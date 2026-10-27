"""Explicit operator actions, never available through the public inference API."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from .registry import ROOT, MODEL_ID, model_manifest


def attach(report_path):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get("protocol") != "ka-heldout-v1" or report.get("testSamples", 0) < 1:
        raise ValueError("Invalid evaluation report")
    count = 0
    for result in report["results"]:
        manifest = model_manifest(result["model"])
        fingerprint = hashlib.sha256(
            json.dumps(manifest["sha256"], sort_keys=True).encode()
        ).hexdigest()
        if fingerprint != result.get("fingerprint"):
            raise ValueError("Evaluation does not match current model artifacts")
        target = manifest.pop("path") / "manifest.json"
        manifest["evaluation"] = {
            "protocol": report["protocol"],
            "createdAt": report["createdAt"],
            "corpusSha256": report["corpusSha256"],
            "fingerprint": fingerprint,
            "environment": report.get("environment"),
            "overall": result["overall"],
            "limitations": report["limitations"],
            "commercialReleaseApproved": False,
        }
        target.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        count += 1
    print(json.dumps({"attached": count, "commercialApproved": False}))


def register(run_path, model_id):
    source = Path(run_path)
    run = json.loads((source / "run.json").read_text())
    if not MODEL_ID.fullmatch(model_id) or not model_id.startswith("ka-"):
        raise ValueError("A ka-* candidate id is required")
    if (
        run.get("kind") != "research-candidate"
        or not run.get("heldout")
        or not run.get("corpusSha256")
    ):
        raise ValueError("Smoke artifacts cannot be registered as a detector")
    threshold = run.get("threshold", 0)
    if not 0 < threshold < 1:
        raise ValueError("Candidate needs a usable held-out threshold")
    target = ROOT / model_id
    if target.exists():
        raise ValueError("Never overwrite a registered model")
    required = [source / "model", source / "calibration.json"]
    if any(not path.exists() or path.is_symlink() for path in required):
        raise ValueError("Missing candidate artifacts")
    target.mkdir()
    try:
        shutil.copytree(source / "model", target / "model")
        shutil.copy2(source / "calibration.json", target / "calibration.json")
        hashes = {}
        for path in sorted(target.rglob("*")):
            if path.is_file():
                with path.open("rb") as file:
                    hashes[str(path.relative_to(target))] = hashlib.file_digest(
                        file, "sha256"
                    ).hexdigest()
        manifest = {
            "id": model_id,
            "status": "local-research",
            "directory": "model",
            "sha256": hashes,
            "threshold": threshold,
            "shortThreshold": threshold,
            "shortWordCutoff": 40,
            "baseModel": run["base"],
            "commercialApproved": False,
            "license": "Candidate: source and corpus rights require review",
            "evaluation": None,
            "training": {
                k: run[k]
                for k in [
                    "baseFingerprint",
                    "seed",
                    "corpusSha256",
                    "createdAt",
                    "device",
                ]
            },
        }
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2))
    except Exception:
        # Only this newly-created, unpublished candidate is removed on a failed copy.
        shutil.rmtree(target)
        raise
    print(
        json.dumps(
            {
                "registeredCandidate": model_id,
                "active": False,
                "evaluationRequired": True,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    attach_parser = sub.add_parser("attach-evaluation")
    attach_parser.add_argument("report")
    register_parser = sub.add_parser("candidate")
    register_parser.add_argument("run")
    register_parser.add_argument("--id", required=True)
    args = parser.parse_args()
    if args.command == "attach-evaluation":
        attach(args.report)
    else:
        register(args.run, args.id)
