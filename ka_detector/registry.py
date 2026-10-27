import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(os.environ.get("KA_MODELS_ROOT", "/models"))
MODEL_ID = re.compile(r"^(upstream-v[1-4]|ka-[a-z0-9-]{1,50})$")
VERIFIED = {}


def model_manifest(model_id, verify=True):
    if not MODEL_ID.fullmatch(model_id):
        raise ValueError("unknown_model")
    directory = ROOT / model_id
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("model_missing")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("id") != model_id or manifest.get("status") not in [
        "local-research",
        "validated",
    ]:
        raise ValueError("model_not_approved")
    if not manifest.get("sha256") or not 0 < manifest.get("threshold", 0) < 1:
        raise ValueError("model_invalid")
    model_path = directory / manifest.get("directory", ".")
    if model_path.is_symlink() or not model_path.resolve().is_relative_to(
        directory.resolve()
    ):
        raise ValueError("artifact_path_invalid")
    if verify:
        stamps = []
        for relative, expected in manifest["sha256"].items():
            path = directory / relative
            if path.is_symlink() or not path.resolve().is_relative_to(
                directory.resolve()
            ):
                raise ValueError("artifact_path_invalid")
            stat = path.stat()
            stamps.append((relative, expected, stat.st_size, stat.st_mtime_ns))
        cache_key = (str(directory.resolve()), json.dumps(stamps))
        if cache_key not in VERIFIED:
            for relative, expected in manifest["sha256"].items():
                path = directory / relative
                with open(path, "rb") as file:
                    actual = hashlib.file_digest(file, "sha256").hexdigest()
                if actual != expected:
                    raise ValueError("model_corrupt")
            VERIFIED[cache_key] = True
    manifest["path"] = directory
    return manifest


def list_models():
    ids = {f"upstream-v{i}" for i in range(1, 5)}
    if ROOT.exists():
        ids.update(
            p.name for p in ROOT.iterdir() if p.is_dir() and MODEL_ID.fullmatch(p.name)
        )
    result = []
    for model_id in sorted(ids):
        try:
            m = model_manifest(model_id)
            result.append(
                {k: v for k, v in m.items() if k not in ["path", "sha256", "directory"]}
                | {
                    "ready": True,
                    "fingerprint": hashlib.sha256(
                        json.dumps(m["sha256"], sort_keys=True).encode()
                    ).hexdigest(),
                }
            )
        except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
            result.append(
                {
                    "id": model_id,
                    "ready": False,
                    "reason": str(error)
                    if isinstance(error, ValueError)
                    else "model_missing",
                    "commercialApproved": False,
                }
            )
    return result
