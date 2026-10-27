import gc
import hashlib
import json
import math
import re
import resource
import time
from collections import OrderedDict
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from aidetector.document_scan import split_paragraphs, classify_sections
from aidetector.features import (
    _words,
    burstiness_and_lengths,
    mattr,
    cliche_rate,
    punctuation_variety,
)
from .registry import model_manifest

MODELS = OrderedDict()
DISCLAIMER = "Исследовательская оценка признаков, не доказательство авторства. Не используйте для санкций, оценки учащихся или кадровых решений."


def load(model_id):
    if model_id in MODELS:
        return MODELS[model_id]
    manifest = model_manifest(model_id)
    if model_id == "upstream-v1":
        raise ValueError("v1_optional_lm_not_prepared")
    while MODELS:
        MODELS.popitem(last=False)
    gc.collect()
    path = manifest["path"] / manifest["directory"]
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            attn_implementation="eager",
        )
        .to("cpu")
        .eval()
    )
    calibration = json.loads((manifest["path"] / "calibration.json").read_text())
    MODELS[model_id] = (manifest, tokenizer, model, calibration)
    return MODELS[model_id]


def warmup(model_id):
    torch.set_num_threads(2)
    _, tokenizer, model, _ = load(model_id)
    with torch.inference_mode():
        model(**tokenizer("Проверка готовности локальной модели.", return_tensors="pt"))
    return {"ready": True, "model": model_id}


def analyze(text, model_ids, diagnostics=False):
    torch.set_num_threads(2)
    started = time.perf_counter()
    words = _words(text)
    paragraphs = split_paragraphs(text)
    split = classify_sections(paragraphs)
    content = split.content
    # A short plain input need not contain a document heading or a title page.
    if not content and not split.dropped:
        content = paragraphs
    content_ids = {p.idx for p in content}
    if not content:
        raise ValueError("no_content_prose")
    stream = "\n\n".join(p.text for p in content)
    nwords = len(_words(stream))
    letters = re.findall(r"[^\W\d_]", stream, re.UNICODE)
    russian = len(re.findall(r"[а-яё]", stream, re.I)) / max(1, len(letters))
    burst, mean_length = burstiness_and_lengths(stream)
    results = []
    for model_id in model_ids:
        before = time.perf_counter()
        if model_id == "upstream-v1":
            import joblib
            from .prepare import digest
            from aidetector import features

            manifest = model_manifest(model_id)
            classifier_path = manifest["path"] / "model.joblib"
            if (
                digest(classifier_path)
                != "dba76ccd76db993977d3000722f90f7a49262e6fc7eecc6d2b806992ff77e808"
            ):
                raise ValueError("v1_classifier_untrusted")
            features.PERPLEXITY_MODEL = str(manifest["path"] / "lm")
            classifier = joblib.load(classifier_path)

            def predict(value):
                return float(
                    classifier.predict_proba(
                        [features.to_vector(features.extract(value))]
                    )[0, 1]
                )

            score = predict(stream)
            mapped = [
                {
                    "index": p.idx,
                    "start": p.start,
                    "end": p.end,
                    "signal": predict(p.text) if len(p.text) >= 200 else None,
                    "short": len(_words(p.text)) < 40,
                }
                for p in content[:96]
            ]
            threshold = manifest["threshold"]
            results.append(
                {
                    "model": model_id,
                    "fingerprint": hashlib.sha256(
                        json.dumps(manifest["sha256"], sort_keys=True).encode()
                    ).hexdigest(),
                    "score": score,
                    "threshold": threshold,
                    "verdict": "insufficient"
                    if nwords < 40 or russian < 0.6
                    else "ambiguous"
                    if abs(score - threshold) < 0.08
                    else "signals"
                    if score >= threshold
                    else "no_strong_signals",
                    "confidence": "independent_calibration_not_established",
                    "aggregation": "upstream_v1_features",
                    "windows": len(mapped),
                    "milliseconds": round((time.perf_counter() - before) * 1000),
                    "paragraphs": mapped,
                    "commercialApproved": False,
                    "evaluation": manifest.get("evaluation"),
                }
            )
            continue
        manifest, tokenizer, model, calibration = load(model_id)
        encoded = tokenizer(
            stream,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        windows = [ids[i : i + 254] for i in range(0, len(ids), 254)]
        if len(windows) > 96:
            raise ValueError("window_limit")
        margins, regions = [], []
        with torch.inference_mode():
            for i in range(0, len(windows), 4):
                # All registered transformer baselines are BERT. Build their
                # CLS/content/SEP batch explicitly; Transformers 5 removed
                # prepare_for_model, and decoding/re-tokenizing would shift offsets.
                if any(
                    value is None
                    for value in [
                        tokenizer.cls_token_id,
                        tokenizer.sep_token_id,
                        tokenizer.pad_token_id,
                    ]
                ):
                    raise ValueError("unsupported_tokenizer")
                group = windows[i : i + 4]
                size = max(len(window) for window in group) + 2
                input_ids = torch.full(
                    (len(group), size), tokenizer.pad_token_id, dtype=torch.long
                )
                attention = torch.zeros_like(input_ids)
                for row, window in enumerate(group):
                    sequence = [tokenizer.cls_token_id, *window, tokenizer.sep_token_id]
                    input_ids[row, : len(sequence)] = torch.tensor(sequence)
                    attention[row, : len(sequence)] = 1
                batch = {
                    "input_ids": input_ids,
                    "attention_mask": attention,
                    "token_type_ids": torch.zeros_like(input_ids),
                }
                logits = model(**batch).logits.numpy()
                for j, pair in enumerate(logits):
                    index = i + j
                    margin = float(pair[1] - pair[0])
                    raw = 1 / (1 + math.exp(-max(-40, min(40, margin))))
                    score = float(np.interp(raw, calibration["x"], calibration["y"]))
                    start = offsets[index * 254][0]
                    end = offsets[min(len(offsets), (index + 1) * 254) - 1][1]
                    margins.append((margin, len(windows[index])))
                    regions.append(
                        {"start": start, "end": end, "score": round(score, 6)}
                    )
        # Aggregate raw evidence over token-disjoint windows, not mean sentence percentages.
        margin = sum(m * n for m, n in margins) / max(1, sum(n for _, n in margins))
        raw = 1 / (1 + math.exp(-max(-40, min(40, margin))))
        score = float(np.interp(raw, calibration["x"], calibration["y"]))
        threshold = (
            manifest["shortThreshold"]
            if nwords < manifest["shortWordCutoff"]
            else manifest["threshold"]
        )
        uncertain = nwords < 40 or russian < 0.6 or abs(score - threshold) < 0.08
        verdict = (
            "insufficient"
            if nwords < 40 or russian < 0.6
            else "ambiguous"
            if uncertain
            else "signals"
            if score >= threshold
            else "no_strong_signals"
        )
        mapped, pos = [], 0
        for paragraph in content:
            end = pos + len(paragraph.text)
            overlapping = [
                r["score"] for r in regions if r["end"] > pos and r["start"] < end
            ]
            mapped.append(
                {
                    "index": paragraph.idx,
                    "start": paragraph.start,
                    "end": paragraph.end,
                    "signal": max(overlapping) if overlapping else None,
                    "short": len(_words(paragraph.text)) < 40,
                }
            )
            pos = end + 2
        fingerprint = hashlib.sha256(
            json.dumps(manifest["sha256"], sort_keys=True).encode()
        ).hexdigest()
        results.append(
            {
                "model": model_id,
                "fingerprint": fingerprint,
                "score": round(score, 6),
                "threshold": threshold,
                "verdict": verdict,
                "confidence": "independent_calibration_not_established",
                "aggregation": "token_weighted_logits_v1"
                if len(windows) > 1
                else "upstream_single_window",
                "windows": len(windows),
                "milliseconds": round((time.perf_counter() - before) * 1000),
                "paragraphs": mapped,
                "commercialApproved": manifest["commercialApproved"],
                "evaluation": manifest.get("evaluation"),
            }
        )
    return {
        "schemaVersion": 1,
        "characters": len(text),
        "words": len(words),
        "sentences": len(list(__import__("razdel").sentenize(text))),
        "statistics": {
            "sentenceLengthStd": round(burst, 3),
            "meanSentenceWords": round(mean_length, 3),
            "lexicalDiversity": round(mattr(words), 4),
            "clichesPer100Words": round(cliche_rate(text, len(words)), 3),
            "punctuationVariety": round(punctuation_variety(text), 3),
        },
        "excludedParagraphs": [p.idx for p in paragraphs if p.idx not in content_ids],
        "results": results,
        "milliseconds": round((time.perf_counter() - started) * 1000),
        "peakMemoryMb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
        ),
        "disclaimer": DISCLAIMER,
        "warnings": [
            "Балл модели не является вероятностью установленного авторства.",
            "Для длинного документа агрегирование логитов экспериментальное; независимая калибровка не установлена.",
            "Подсветка переносится с контекстных окон, а не доказывает происхождение отдельного предложения.",
        ],
    }
