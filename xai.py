"""TensorFlow Integrated Gradients XAI visualization."""
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image, ImageOps
from typing import Tuple

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


class TensorFlowXAIVisualizer:
    """Integrated-Gradients-based XAI visualizer for Teachable Machine models."""

    def __init__(self, model: tf.keras.Model) -> None:
        self.model = model

    def generate_explanation(
        self, image: np.ndarray, class_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (overlay_bgr, heatmap_display) for the given BGR image."""
        img_array = self._preprocess_image(image)
        try:
            return self._generate_integrated_gradients(img_array, image, class_idx)
        except Exception:  # noqa: BLE001 - fall back to a cheaper method
            return self._generate_guided_backprop(img_array, image, class_idx)

    def _generate_integrated_gradients(
        self, img_array: tf.Tensor, original_image: np.ndarray, class_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate an IG heatmap and return (overlay_bgr, heatmap_display)."""
        baseline = tf.zeros_like(img_array[0])
        heatmap = integrated_gradients(
            self.model, img_array[0].numpy(), baseline.numpy(), class_idx
        )
        heatmap = self._smooth_heatmap(heatmap, sigma=2.0)
        return self._create_visualization(original_image, heatmap)

    def _generate_guided_backprop(
        self, img_array: tf.Tensor, original_image: np.ndarray, class_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback: gradient-magnitude saliency when IG fails."""
        with tf.GradientTape() as tape:
            tape.watch(img_array)
            predictions = self.model(img_array)
            class_output = predictions[0][class_idx]
        grads = tape.gradient(class_output, img_array)
        if grads is None:
            raise ValueError("Could not compute gradients")
        grads = tf.abs(grads[0])
        importance = tf.reduce_mean(grads, axis=-1)
        importance = (importance - tf.reduce_min(importance)) / (
            tf.reduce_max(importance) - tf.reduce_min(importance) + 1e-8
        )
        heatmap = self._smooth_heatmap(importance.numpy(), sigma=2.0)
        return self._create_visualization(original_image, heatmap)

    def _smooth_heatmap(self, heatmap: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian smoothing to a heatmap (scipy, OpenCV fallback)."""
        try:
            from scipy.ndimage import gaussian_filter

            return gaussian_filter(heatmap, sigma=sigma)
        except ImportError:
            kernel = max(3, int(2 * np.ceil(2 * sigma) + 1))
            if kernel % 2 == 0:
                kernel += 1
            return cv2.GaussianBlur(heatmap, (kernel, kernel), sigma)

    def _create_visualization(
        self, image: np.ndarray, heatmap: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Overlay a JET-colored heatmap on the center-cropped image."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        pil_image = Image.fromarray(image_rgb.astype(np.uint8))
        pil_cropped = ImageOps.fit(
            pil_image, (config.IMG_SIZE, config.IMG_SIZE), Image.Resampling.LANCZOS
        )
        cropped = np.asarray(pil_cropped)
        rgb_img = cropped.astype(np.float32) / 255.0

        heatmap_smooth = cv2.GaussianBlur(heatmap, (11, 11), 2.0)
        heatmap_enhanced = np.clip(np.power(heatmap_smooth, 0.7), 0, 1)
        colored_bgr = cv2.applyColorMap(
            np.uint8(255 * heatmap_enhanced), cv2.COLORMAP_JET
        )
        colored = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        rgb_uint8 = (rgb_img * 255).astype(np.uint8)
        colored_uint8 = (colored * 255).astype(np.uint8)
        alpha = 0.6
        overlay = cv2.addWeighted(rgb_uint8, 1 - alpha, colored_uint8, alpha, 0)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        heatmap_display = np.uint8(255 * heatmap)
        return overlay_bgr, heatmap_display

    def _preprocess_image(self, image: np.ndarray) -> tf.Tensor:
        """Preprocess a BGR image into a (1,224,224,3) normalized tensor."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        pil_image = Image.fromarray(image_rgb.astype(np.uint8))
        pil_image = ImageOps.fit(
            pil_image, (config.IMG_SIZE, config.IMG_SIZE), Image.Resampling.LANCZOS
        )
        image_array = np.asarray(pil_image)
        normalized = (image_array.astype(np.float32) / 127.5) - 1
        img_array = np.expand_dims(normalized, axis=0)
        return tf.convert_to_tensor(img_array)
