"""Operator-only, reproducible evaluation/training; never consumes visitor documents."""

import argparse
import hashlib
import json
import importlib.metadata
import platform
import math
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np


def load_corpus(path):
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = {
        "id",
        "text",
        "label",
        "genre",
        "language",
        "source",
        "sourceGroup",
        "license",
        "provenance",
        "generator",
        "edited",
        "addedAt",
        "split",
    }
    ids, texts, groups = set(), {}, {}
    for row in rows:
        if (
            not required.issubset(row)
            or row["label"] not in ["human", "ai", "mixed"]
            or row["split"] not in ["train", "validation", "test"]
            or not row["license"]
            or not row["provenance"]
            or row["language"] != "ru"
        ):
            raise ValueError(
                "Every sample needs explicit provenance, license and split"
            )
        if row["id"] in ids:
            raise ValueError("Duplicate sample id")
        ids.add(row["id"])
        normalized = re.sub(r"\W+", " ", row["text"].lower().replace("ё", "е")).strip()
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        if digest in texts:
            raise ValueError("Exact duplicate text")
        if row["sourceGroup"] in groups and groups[row["sourceGroup"]] != row["split"]:
            raise ValueError("Source-group leakage across splits")
        groups[row["sourceGroup"]] = row["split"]
        for other in texts.values():
            if (
                other["split"] != row["split"]
                and SequenceMatcher(
                    None, normalized, other["normalized"], autojunk=False
                ).ratio()
                > 0.85
            ):
                raise ValueError("Near-duplicate leakage across splits")
        texts[digest] = {"split": row["split"], "normalized": normalized}
    return rows


def interval(k, n):
    if not n:
        return None
    z = 1.95996398454
    center = (k / n + z * z / (2 * n)) / (1 + z * z / n)
    half = (
        z * math.sqrt((k / n) * (1 - k / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    )
    return [max(0, center - half), min(1, center + half)]


def metrics(rows):
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    binary = [r for r in rows if r["label"] != "mixed"]
    if not binary:
        return {"n": 0}
    y = [int(r["label"] == "ai") for r in binary]
    p = [r["predicted"] for r in binary]
    h = [r for r in binary if r["label"] == "human"]
    a = [r for r in binary if r["label"] == "ai"]
    fp = sum(r["predicted"] for r in h)
    fn = sum(not r["predicted"] for r in a)
    return {
        "n": len(binary),
        "human": len(h),
        "ai": len(a),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "rocAuc": float(roc_auc_score(y, [r["score"] for r in binary]))
        if len(set(y)) == 2
        else None,
        "fpr": fp / len(h) if h else None,
        "fnr": fn / len(a) if a else None,
        "fprWilson95": interval(fp, len(h)),
        "fnrWilson95": interval(fn, len(a)),
        "milliseconds": sum(r["milliseconds"] for r in binary) / len(binary),
        "peakMemoryMb": max(r.get("memoryMb", 0) for r in binary),
    }


def evaluate(path, models, output):
    from .engine import analyze

    rows = load_corpus(path)
    samples = [r for r in rows if r["split"] == "test"]
    all_results = []
    for model in models:
        from .registry import model_manifest

        manifest = model_manifest(model)
        fingerprint = hashlib.sha256(
            json.dumps(manifest["sha256"], sort_keys=True).encode()
        ).hexdigest()
        observations = []
        for row in samples:
            result = analyze(row["text"], [model])
            signal = result["results"][0]
            observations.append(
                {
                    "id": row["id"],
                    "label": row["label"],
                    "genre": row["genre"],
                    "generator": row["generator"],
                    "words": result["words"],
                    "score": signal["score"],
                    "predicted": signal["score"] >= signal["threshold"],
                    "milliseconds": result["milliseconds"],
                    "memoryMb": result["peakMemoryMb"],
                }
            )
        overall = metrics(observations)
        all_results.append(
            {
                "model": model,
                "fingerprint": fingerprint,
                "overall": overall,
                "byGenre": {
                    g: metrics([r for r in observations if r["genre"] == g])
                    for g in sorted({r["genre"] for r in observations})
                },
                "byGenerator": {
                    g: metrics([r for r in observations if r["generator"] == g])
                    for g in sorted({str(r["generator"]) for r in observations})
                },
                "byLength": {
                    "short": metrics([r for r in observations if r["words"] < 40]),
                    "long": metrics([r for r in observations if r["words"] >= 40]),
                },
                "observations": observations,
            }
        )
    report = {
        "protocol": "ka-heldout-v1",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ["torch", "transformers", "numpy", "scikit-learn"]
            },
        },
        "corpusSha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "testSamples": len(samples),
        "results": all_results,
        "limitations": [
            "Small research corpus, not representative product validation.",
            "Upstream complete training corpus is unavailable: independence from upstream fine-tuning cannot be proven.",
            "Wilson intervals are descriptive; correlated source fragments reduce effective sample size.",
            "Do not claim FPR <= 3% from a small point estimate.",
        ],
        "commercialReleaseApproved": False,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({"report": str(output), "samples": len(samples), "models": models})
    )


def train(path, output, model_id="upstream-v2", smoke=False, device="cpu", epochs=1):
    import torch
    from .engine import load

    torch.set_num_threads(2)
    torch.manual_seed(27)
    np.random.seed(27)
    random.seed(27)
    manifest, tokenizer, model, _ = load(model_id)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA is not available in this image; use the explicit GPU training profile"
        )
    model.to(device).train()
    if smoke:
        rows = [
            {
                "text": "Технический smoke-test градиента и сохранения, не обучающий корпус авторства. "
                * 4,
                "label": "ai",
            }
        ]
    else:
        rows = load_corpus(path)
        groups = Counter(r["split"] for r in rows if r["label"] != "mixed")
        if groups["train"] < 200 or groups["validation"] < 40 or groups["test"] < 60:
            raise ValueError(
                "Insufficient approved data; full training was not started"
            )
        if any(
            len(
                {
                    r["label"]
                    for r in rows
                    if r["split"] == split and r["label"] != "mixed"
                }
            )
            < 2
            for split in ["train", "validation", "test"]
        ):
            raise ValueError("Both labels are required in every split")
        rows = [r for r in rows if r["split"] == "train" and r["label"] != "mixed"]
    target = Path(output)
    if target.exists():
        raise ValueError("Run directory already exists; never overwrite an earlier run")
    target.mkdir(parents=True)
    started = time.perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)
    losses = []
    for _ in range(min(epochs, 2)):
        for index in range(0, len(rows), 2):
            if time.perf_counter() - started > 600:
                raise ValueError("Training budget exceeded; run is incomplete")
            batch = rows[index : index + 2]
            encoded = tokenizer(
                [r["text"] for r in batch],
                max_length=128,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            labels = torch.tensor(
                [int(r["label"] == "ai") for r in batch], device=device
            )
            optimizer.zero_grad()
            loss = model(**encoded, labels=labels).loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    model.save_pretrained(target / "model", safe_serialization=True)
    tokenizer.save_pretrained(target / "model")
    run = {
        "kind": "synthetic-gradient-smoke" if smoke else "research-candidate",
        "base": model_id,
        "baseFingerprint": hashlib.sha256(
            json.dumps(manifest["sha256"], sort_keys=True).encode()
        ).hexdigest(),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "epochs": min(epochs, 2),
        "seed": 27,
        "losses": losses,
        "milliseconds": round((time.perf_counter() - started) * 1000),
        "torch": torch.__version__,
        "corpusSha256": None
        if smoke
        else hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "usableDetector": False,
        "automaticPromotion": False,
    }
    if not smoke:
        from sklearn.isotonic import IsotonicRegression

        model.eval()
        source = load_corpus(path)

        def score_rows(split):
            selected = [
                r for r in source if r["split"] == split and r["label"] != "mixed"
            ]
            scores = []
            with torch.inference_mode():
                for r in selected:
                    enc = tokenizer(
                        r["text"], max_length=256, truncation=True, return_tensors="pt"
                    ).to(device)
                    scores.append(
                        float(torch.softmax(model(**enc).logits, dim=1)[0, 1].cpu())
                    )
            return selected, scores

        validation, scores = score_rows("validation")
        y = [int(r["label"] == "ai") for r in validation]
        calibration = IsotonicRegression(out_of_bounds="clip").fit(scores, y)
        (target / "calibration.json").write_text(
            json.dumps(
                {
                    "x": calibration.X_thresholds_.tolist(),
                    "y": calibration.y_thresholds_.tolist(),
                }
            )
        )
        adjusted = calibration.predict(scores)
        human = [s for s, r in zip(adjusted, validation) if r["label"] == "human"]
        threshold = next(
            (
                float(t)
                for t in sorted(set(adjusted))
                if sum(s >= t for s in human) / len(human) <= 0.03
            ),
            1.0,
        )
        test, raw = score_rows("test")
        observed = [
            {
                "label": r["label"],
                "score": float(s),
                "predicted": bool(s >= threshold),
                "milliseconds": 0,
            }
            for r, s in zip(test, calibration.predict(raw))
        ]
        run.update(
            {
                "threshold": threshold,
                "heldout": metrics(observed),
                "approval": "manual-review-required; not registered with inference",
            }
        )
    (target / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps(run))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    validator = sub.add_parser("validate")
    validator.add_argument("corpus")
    evaluator = sub.add_parser("evaluate")
    evaluator.add_argument("corpus")
    evaluator.add_argument(
        "--models", nargs="+", default=["upstream-v2", "upstream-v4"]
    )
    evaluator.add_argument("--output", default="/runs/evaluation.json")
    trainer = sub.add_parser("train")
    trainer.add_argument("corpus")
    trainer.add_argument("--output", required=True)
    trainer.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    trainer.add_argument("--epochs", type=int, choices=[1, 2], default=1)
    smoke = sub.add_parser("train-smoke")
    smoke.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "validate":
        print(json.dumps({"valid": True, "samples": len(load_corpus(args.corpus))}))
    elif args.command == "evaluate":
        evaluate(args.corpus, args.models, args.output)
    elif args.command == "train-smoke":
        train(None, args.output, smoke=True)
    else:
        train(args.corpus, args.output, device=args.device, epochs=args.epochs)
