"""Explicit network-enabled preparation; never called by inference."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REPO = "k1y0mi/ru-ai-text-detector"
REVISION = "175e75ec79e9471f3f5b4ab0da9b859188446807"
SOURCE_COMMIT = "244e955427548f5856bb389cc8611f959911290c"
EXPECTED = {
    "v2": "c960c584fb4e2086d81cff7516f8620ba559f5f05ea6572b6b68243e06f04a1a",
    "v3": "12156dba0fc8102f3bb204eaaf2230d03302eaa342cd6b929ae08ec971e1ee9f",
    "v4": "3471644dc97d149c368286d0a7779bafdcbe4237f19226b29d75eb7e767d2a1d",
}
CALIBRATORS = {
    "v2": "238ed399da278e35702581a2361552b4de5526160624d3a57739d7af794652c7",
    "v3": "6cc5835d53764b267167f5a7d258ac150084c1707c50f1d54db56e5a0c7c85c3",
    "v4": "13d841343ee759e63ba020593f0b8dd67a8559abfe89b4c4f76f627695b09002",
}


def digest(path):
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def prepare(root, versions):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for version in versions:
        if version not in EXPECTED:
            raise ValueError("Unsupported pinned upstream version")
        directory = root / f"upstream-{version}"
        directory.mkdir(exist_ok=True)
        names = [
            f"{version}_model/{f}"
            for f in [
                "config.json",
                "model.safetensors",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.txt",
            ]
        ]
        names += [f"{version}_calibrator.joblib", f"threshold_{version}.json"]
        hashes = {}
        for name in names:
            target = directory / name
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                temporary = target.with_suffix(target.suffix + ".part")
                with (
                    urlopen(
                        f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}",
                        timeout=90,
                    ) as response,
                    open(temporary, "wb") as output,
                ):
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                temporary.replace(target)
            hashes[name] = digest(target)
            if name.endswith("model.safetensors") and hashes[name] != EXPECTED[version]:
                raise ValueError("Pinned model checksum mismatch")
        # The pickle is loaded only here, from the reviewed immutable upstream source.
        # Inference receives a non-executable JSON calibration table instead.
        import joblib

        if hashes[f"{version}_calibrator.joblib"] != CALIBRATORS[version]:
            raise ValueError("Calibrator differs from reviewed Git artifact")
        calibrator = joblib.load(directory / f"{version}_calibrator.joblib")
        calibration = {
            "x": calibrator.X_thresholds_.tolist(),
            "y": calibrator.y_thresholds_.tolist(),
        }
        (directory / "calibration.json").write_text(
            json.dumps(calibration), encoding="utf-8"
        )
        hashes["calibration.json"] = digest(directory / "calibration.json")
        threshold = json.loads(
            (directory / f"threshold_{version}.json").read_text(encoding="utf-8")
        )
        manifest = {
            "id": f"upstream-{version}",
            "origin": f"https://huggingface.co/{REPO}/tree/{REVISION}",
            "sourceCommit": SOURCE_COMMIT,
            "revision": REVISION,
            "baseModel": threshold.get("base", "ai-forever/ruBert-base"),
            "directory": f"{version}_model",
            "threshold": threshold["threshold"],
            "shortThreshold": threshold.get("threshold_short", threshold["threshold"]),
            "shortWordCutoff": threshold.get("short_word_cutoff", 0),
            "maxTokens": 256,
            "status": "local-research",
            "license": "unconfirmed-finetune",
            "commercialApproved": False,
            "trainingDate": None,
            "corpusVersion": "upstream-not-redistributed",
            "sha256": hashes,
            "evaluation": None,
        }
        temporary = directory / "manifest.json.next"
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(directory / "manifest.json")
        print(
            json.dumps(
                {
                    "prepared": manifest["id"],
                    "files": len(hashes),
                    "commercialApproved": False,
                }
            )
        )


def prepare_v1(root):
    root = Path(root) / "upstream-v1"
    root.mkdir(parents=True, exist_ok=True)
    repo, revision = (
        "ai-forever/rugpt3small_based_on_gpt2",
        "a9307e696cd3c5b7f953ff4cb19d76a4d81821d5",
    )
    hashes = {}
    for name in [
        "config.json",
        "merges.txt",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json",
        "model.joblib",
    ]:
        target = root / (name if name == "model.joblib" else f"lm/{name}")
        target.parent.mkdir(exist_ok=True)
        url = (
            f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
            if name != "model.joblib"
            else f"https://raw.githubusercontent.com/k1y0miiii/ru-ai-text-detector/{SOURCE_COMMIT}/models/model.joblib"
        )
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".part")
            with urlopen(url, timeout=90) as response, open(temporary, "wb") as file:
                while chunk := response.read(1024 * 1024):
                    file.write(chunk)
            temporary.replace(target)
        hashes[str(target.relative_to(root))] = digest(target)
    if (
        hashes["lm/pytorch_model.bin"]
        != "0d2a4a387b0430654bb1f6813436bd31f8e10a06e18ea0e5c41b33120b78458e"
        or hashes["model.joblib"]
        != "dba76ccd76db993977d3000722f90f7a49262e6fc7eecc6d2b806992ff77e808"
    ):
        raise ValueError("v1 immutable artifact mismatch")
    manifest = {
        "id": "upstream-v1",
        "origin": f"https://github.com/k1y0miiii/ru-ai-text-detector/tree/{SOURCE_COMMIT}",
        "revision": revision,
        "baseModel": repo,
        "threshold": 0.86,
        "status": "local-research",
        "license": "unconfirmed-model-and-corpus",
        "commercialApproved": False,
        "sha256": hashes,
        "evaluation": None,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"prepared": "upstream-v1", "commercialApproved": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/models")
    parser.add_argument("--versions", nargs="+", default=["v2"])
    parser.add_argument("--with-v1", action="store_true")
    args = parser.parse_args()
    prepare(args.root, args.versions)
    if args.with_v1:
        prepare_v1(args.root)
