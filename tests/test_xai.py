"""Tests for the Integrated Gradients implementation."""
import numpy as np
import tensorflow as tf

import config
from xai import integrated_gradients


def _reference_importance(model, image, baseline, class_idx, m_steps):
    """All-at-once reference implementation (original behavior)."""
    image = tf.convert_to_tensor(image, tf.float32)
    baseline = tf.convert_to_tensor(baseline, tf.float32)
    alphas = tf.linspace(0.0, 1.0, m_steps + 1)
    a = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    delta = image - baseline
    imgs = baseline[tf.newaxis, ...] + a * delta[tf.newaxis, ...]
    with tf.GradientTape() as tape:
        tape.watch(imgs)
        preds = model(imgs)
        outputs = preds[:, class_idx]
    grads = tape.gradient(outputs, imgs)
    avg = tf.reduce_mean(grads, axis=0)
    integrated = (image - baseline) * avg
    importance = tf.reduce_mean(tf.abs(integrated), axis=-1)
    importance = (importance - tf.reduce_min(importance)) / (
        tf.reduce_max(importance) - tf.reduce_min(importance) + 1e-8
    )
    return importance.numpy()


def test_integrated_gradients_batched_matches_unbatched(tiny_model, sample_image):
    baseline = np.zeros_like(sample_image)
    ref = _reference_importance(tiny_model, sample_image, baseline, 1, m_steps=20)
    got = integrated_gradients(
        tiny_model, sample_image, baseline, class_idx=1, m_steps=20, batch_size=4
    )
    assert got.shape == (config.IMG_SIZE, config.IMG_SIZE)
    assert np.allclose(ref, got, atol=1e-4)


def test_integrated_gradients_output_normalized(tiny_model, sample_image):
    baseline = np.zeros_like(sample_image)
    got = integrated_gradients(tiny_model, sample_image, baseline, class_idx=0)
    assert got.min() >= 0.0
    assert got.max() <= 1.0 + 1e-6


def test_visualizer_generate_explanation_shapes(tiny_model):
    from xai import TensorFlowXAIVisualizer

    viz = TensorFlowXAIVisualizer(tiny_model)
    image_bgr = (np.random.RandomState(1).rand(300, 400, 3) * 255).astype(np.uint8)
    overlay, heatmap = viz.generate_explanation(image_bgr, class_idx=0)
    assert overlay.shape == (config.IMG_SIZE, config.IMG_SIZE, 3)
    assert overlay.dtype == np.uint8
    assert heatmap.shape == (config.IMG_SIZE, config.IMG_SIZE)
    assert heatmap.dtype == np.uint8


def test_registry_lists_default_first():
    from xai import list_xai_methods
    methods = list_xai_methods()
    assert methods[0] == "Integrated Gradients"
    assert "Integrated Gradients" in methods


def test_get_xai_method_falls_back_to_default():
    from xai import get_xai_method, XAI_METHODS
    assert callable(get_xai_method("Integrated Gradients"))
    assert get_xai_method("Nonexistent") is XAI_METHODS["Integrated Gradients"]


def test_ig_adapter_returns_normalized_map(tiny_model, sample_image):
    from xai import get_xai_method
    fn = get_xai_method("Integrated Gradients")
    hm = fn(tiny_model, sample_image, 0)
    assert hm.shape == (config.IMG_SIZE, config.IMG_SIZE)
    assert hm.min() >= 0.0 and hm.max() <= 1.0 + 1e-6


def test_rise_shape_and_range(tiny_model, sample_image):
    from xai import rise
    hm = rise(tiny_model, sample_image, 0, n_masks=64, batch_size=32)
    assert hm.shape == (config.IMG_SIZE, config.IMG_SIZE)
    assert hm.dtype == np.float32
    assert hm.min() >= 0.0 and hm.max() <= 1.0 + 1e-6


def test_rise_deterministic_with_seed(tiny_model, sample_image):
    from xai import rise
    a = rise(tiny_model, sample_image, 0, n_masks=64, batch_size=32, seed=123)
    b = rise(tiny_model, sample_image, 0, n_masks=64, batch_size=32, seed=123)
    assert np.allclose(a, b)


def test_rise_registered():
    from xai import list_xai_methods
    assert "RISE" in list_xai_methods()


def test_rise_distinct_seeds_differ(tiny_model, sample_image):
    from xai import rise
    a = rise(tiny_model, sample_image, 0, n_masks=64, batch_size=32, seed=123)
    c = rise(tiny_model, sample_image, 0, n_masks=64, batch_size=32, seed=999)
    assert not np.allclose(a, c)
