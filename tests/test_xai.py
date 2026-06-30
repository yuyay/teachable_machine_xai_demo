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
