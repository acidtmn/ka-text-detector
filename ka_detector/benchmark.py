"""Apples-to-apples single-window latency and independent-process RSS."""

import json
import statistics
import sys
import time
import psutil

TEXT = (
    "В мастерской заменили старый кабель. После проверки оказалось, что разъём плохо держался в корпусе. Мы записали результат и подготовили запасную деталь. "
    * 4
)


def main(mode):
    import torch

    torch.set_num_threads(2)
    if mode == "original":
        from aidetector import features, transformer_score as tf
        from .registry import ROOT

        features.PERPLEXITY_MODEL = str(ROOT / "upstream-v1/lm")
        root = ROOT / "upstream-v2"
        tf.REGISTRY["benchmark"] = {
            "model": str(root / "v2_model"),
            "calib": str(root / "v2_calibrator.joblib"),
            "thr": str(root / "threshold_v2.json"),
        }
        scorer = tf.Scorer("benchmark")
        scorer._cache_path = "/tmp/ka-benchmark-cache.json"

        def run():
            features.extract(TEXT)
            scorer._cache = {}
            return float(scorer.proba([TEXT])[0])
    else:
        from .engine import analyze

        def run():
            return analyze(TEXT, ["upstream-v2"])["results"][0]["score"]

    first = time.perf_counter()
    score = run()
    cold = (time.perf_counter() - first) * 1000
    times = []
    for _ in range(3):
        before = time.perf_counter()
        run()
        times.append((time.perf_counter() - before) * 1000)
    print(
        json.dumps(
            {
                "mode": mode,
                "score": score,
                "coldMs": round(cold, 2),
                "medianWarmMs": round(statistics.median(times), 2),
                "rssMb": round(psutil.Process().memory_info().rss / 1024**2, 1),
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1])
