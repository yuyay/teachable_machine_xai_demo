"""End-to-end pipeline smoke test (camera-independent)."""
import numpy as np

import config
from model import load_model_from_zip
from xai import TensorFlowXAIVisualizer


def test_full_pipeline_predict_then_explain(model_zip_bytes):
    tm = load_model_from_zip(model_zip_bytes)
    image_bgr = (np.random.RandomState(3).rand(480, 640, 3) * 255).astype(np.uint8)

    probs, idx = tm.predict(image_bgr)
    assert 0 <= idx < probs.shape[0]

    viz = TensorFlowXAIVisualizer(tm.model)
    overlay, heatmap = viz.generate_explanation(image_bgr, idx)
    assert overlay.shape == (config.IMG_SIZE, config.IMG_SIZE, 3)
    assert heatmap.shape == (config.IMG_SIZE, config.IMG_SIZE)
