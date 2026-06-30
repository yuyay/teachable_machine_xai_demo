"""TensorFlow Integrated Gradients XAI visualization."""
import numpy as np
import tensorflow as tf

import config


def integrated_gradients(
    model: tf.keras.Model,
    image: np.ndarray,
    baseline: np.ndarray,
    class_idx: int,
    m_steps: int = config.M_STEPS,
    batch_size: int = config.IG_BATCH,
) -> np.ndarray:
    """Compute a normalized Integrated Gradients importance map.

    Processes the (m_steps + 1) interpolation images in mini-batches to bound
    peak memory, accumulating gradients instead of recording one giant tape.
    Numerically equivalent to the all-at-once mean of per-step gradients.

    Args:
        model: A Keras model mapping (N, H, W, 3) -> (N, num_classes).
        image: Preprocessed input image, shape (H, W, 3), float32.
        baseline: Baseline image (e.g. zeros), same shape as image.
        class_idx: Target class index.
        m_steps: Number of interpolation steps.
        batch_size: Interpolation images per gradient batch.

    Returns:
        Normalized importance map of shape (H, W), float32 in [0, 1].
    """
    image_t = tf.convert_to_tensor(image, dtype=tf.float32)
    baseline_t = tf.convert_to_tensor(baseline, dtype=tf.float32)
    alphas = tf.linspace(0.0, 1.0, m_steps + 1)
    num = m_steps + 1
    delta = image_t - baseline_t

    grad_sum = tf.zeros_like(image_t)
    for start in range(0, num, batch_size):
        batch_alphas = alphas[start:start + batch_size]
        a = batch_alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
        imgs = baseline_t[tf.newaxis, ...] + a * delta[tf.newaxis, ...]
        with tf.GradientTape() as tape:
            tape.watch(imgs)
            preds = model(imgs)
            outputs = preds[:, class_idx]
        grads = tape.gradient(outputs, imgs)
        grad_sum += tf.reduce_sum(grads, axis=0)

    avg_grads = grad_sum / tf.cast(num, tf.float32)
    integrated = delta * avg_grads
    importance = tf.reduce_mean(tf.abs(integrated), axis=-1)
    importance = (importance - tf.reduce_min(importance)) / (
        tf.reduce_max(importance) - tf.reduce_min(importance) + 1e-8
    )
    return importance.numpy()
