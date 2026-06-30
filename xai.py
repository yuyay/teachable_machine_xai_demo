"""TensorFlow Integrated Gradients XAI visualization."""
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image, ImageOps
from typing import Callable, Dict, List, Tuple

import config


XAI_METHODS: Dict[str, Callable[..., np.ndarray]] = {}


def register_xai_method(name: str) -> Callable:
    """Register an XAI method under `name` in the global registry."""
    def deco(fn: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        XAI_METHODS[name] = fn
        return fn
    return deco


def get_xai_method(name: str) -> Callable[..., np.ndarray]:
    """Return the registered method, or the default method for unknown names."""
    return XAI_METHODS.get(name, XAI_METHODS[config.DEFAULT_XAI_METHOD])


def list_xai_methods() -> List[str]:
    """Return registered method names with the default method first."""
    names = list(XAI_METHODS.keys())
    names.sort(key=lambda n: (n != config.DEFAULT_XAI_METHOD, n))
    return names


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


@register_xai_method("Integrated Gradients")
def _ig_explain(
    model: tf.keras.Model, image: np.ndarray, class_idx: int
) -> np.ndarray:
    """Registry adapter: Integrated Gradients with a zero baseline."""
    baseline = np.zeros_like(image)
    return integrated_gradients(model, image, baseline, class_idx)


@register_xai_method("RISE")
def rise(
    model: tf.keras.Model,
    image: np.ndarray,
    class_idx: int,
    n_masks: int = config.RISE_N,
    grid: int = config.RISE_GRID,
    prob: float = config.RISE_PROB,
    batch_size: int = config.RISE_BATCH,
    seed: int = config.RISE_SEED,
) -> np.ndarray:
    """Compute a RISE saliency map via randomized input masking.

    Generates n_masks low-res (grid x grid) binary masks, bilinearly upsamples
    and randomly shifts each to (H, W), occludes the input, and weights each
    mask by the model's score for class_idx. Masks are processed in batches of
    batch_size to bound peak memory. Seeded for reproducibility.

    Args:
        model: Keras model mapping (N, H, W, 3) -> (N, num_classes).
        image: Preprocessed input image, shape (H, W, 3), float32.
        class_idx: Target class index.
        n_masks: Number of random masks.
        grid: Low-resolution mask grid size (grid x grid).
        prob: Probability a grid cell is on.
        batch_size: Masks per model forward batch.
        seed: RNG seed for reproducibility.

    Returns:
        Normalized importance map of shape (H, W), float32 in [0, 1].
    """
    rng = np.random.RandomState(seed)
    h = w = config.IMG_SIZE
    cell_h = int(np.ceil(h / grid))
    cell_w = int(np.ceil(w / grid))
    up_h = (grid + 1) * cell_h
    up_w = (grid + 1) * cell_w
    image = image.astype(np.float32)

    weighted = np.zeros((h, w), dtype=np.float32)
    for start in range(0, n_masks, batch_size):
        b = min(batch_size, n_masks - start)
        grid_masks = (rng.rand(b, grid, grid) < prob).astype(np.float32)
        masks = np.empty((b, h, w), dtype=np.float32)
        masked = np.empty((b, h, w, 3), dtype=np.float32)
        for i in range(b):
            up = cv2.resize(
                grid_masks[i], (up_w, up_h), interpolation=cv2.INTER_LINEAR
            )
            x = rng.randint(0, up_w - w + 1)
            y = rng.randint(0, up_h - h + 1)
            m = up[y:y + h, x:x + w]
            masks[i] = m
            masked[i] = image * m[..., np.newaxis]
        preds = model(masked, training=False).numpy()
        scores = preds[:, class_idx]
        weighted += np.tensordot(scores, masks, axes=([0], [0]))

    saliency = weighted / (float(n_masks) * prob)
    saliency = (saliency - saliency.min()) / (
        saliency.max() - saliency.min() + 1e-8
    )
    return saliency.astype(np.float32)


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
