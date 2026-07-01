"""Layer-1 load test: concurrent in-process predict + IG/RISE.

Run explicitly (NOT part of the default `pytest tests/` suite):
    .venv/bin/python -m pytest loadtest/test_concurrency_compute.py -v -s

Stresses the heavy XAI path under N concurrent threads to confirm it does not
crash, deadlock, or leak. Calls generate_explanation directly, which does NOT
go through app.py's _XAI_SEMAPHORE, so this is a conservative worst case (all N
XAI at once) vs production (semaphore caps concurrent XAI at 2). Absolute RSS
differs from the Cloud Run instance; the soft RSS limit is configurable and the
hard gate is: no exceptions + all workers complete.
"""
import os
import threading
import time

import numpy as np
import psutil

import config
from model import load_model_from_zip
from xai import TensorFlowXAIVisualizer

N_THREADS = int(os.environ.get("LOAD_N", "10"))
ITERATIONS = int(os.environ.get("LOAD_ITERS", "1"))
RSS_LIMIT_GB = float(os.environ.get("LOAD_RSS_LIMIT_GB", "6.0"))


def _sample_peak_rss(stop_event: threading.Event, result: dict) -> None:
    """Background sampler: record the process's peak RSS until stopped."""
    try:
        proc = psutil.Process()
        peak = 0
        while not stop_event.is_set():
            peak = max(peak, proc.memory_info().rss)
            time.sleep(0.1)
        result["peak"] = peak
    except Exception as exc:  # noqa: BLE001 - surface sampler failure, don't report a false 0
        result["peak"] = -1
        result["sampler_error"] = repr(exc)


def test_concurrent_predict_and_explain(model_zip_bytes):
    tm = load_model_from_zip(model_zip_bytes)
    viz = TensorFlowXAIVisualizer(tm.model)
    rng = np.random.RandomState(0)
    images = [
        (rng.rand(240, 320, 3) * 255).astype(np.uint8) for _ in range(N_THREADS)
    ]

    errors: list = []

    def worker(img: np.ndarray) -> None:
        try:
            for _ in range(ITERATIONS):
                _probs, idx = tm.predict(img)
                for method in ("Integrated Gradients", "RISE"):
                    overlay, heatmap = viz.generate_explanation(img, idx, method)
                    assert overlay.shape == (config.IMG_SIZE, config.IMG_SIZE, 3)
                    assert heatmap.shape == (config.IMG_SIZE, config.IMG_SIZE)
        except Exception as exc:  # noqa: BLE001 - collected for the assertion below
            errors.append(repr(exc))

    stop = threading.Event()
    result: dict = {}
    sampler = threading.Thread(
        target=_sample_peak_rss, args=(stop, result), daemon=True
    )
    sampler.start()

    threads = [threading.Thread(target=worker, args=(img,)) for img in images]
    start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - start
    stop.set()
    sampler.join()

    peak = result.get("peak", -1)
    assert peak >= 0, f"RSS sampler failed: {result.get('sampler_error')}"
    peak_gb = peak / 1e9
    print(
        f"\n[load] N={N_THREADS} iters={ITERATIONS} "
        f"elapsed={elapsed:.1f}s peak_RSS={peak_gb:.2f}GB"
    )
    assert not errors, f"{len(errors)}/{N_THREADS} workers raised: {errors[:3]}"
    assert peak_gb < RSS_LIMIT_GB, (
        f"peak RSS {peak_gb:.2f}GB exceeded soft limit {RSS_LIMIT_GB}GB "
        f"(tune via LOAD_RSS_LIMIT_GB)"
    )
