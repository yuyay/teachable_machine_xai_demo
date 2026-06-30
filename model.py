"""Teachable Machine model loading, caching, and inference."""
import hashlib
import os
import tempfile
import zipfile
from typing import List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

import config


def parse_class_names(labels_text: str) -> List[str]:
    """Parse labels.txt content (e.g. '0 iPhone') into class names."""
    names: List[str] = []
    for line in labels_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        names.append(parts[1] if len(parts) >= 2 else parts[0])
    return names


class TeachableMachineModel:
    """Wraps an already-loaded Keras model with TM-style preprocessing."""

    def __init__(self, model, class_names: Optional[List[str]] = None) -> None:
        self.model = model
        self.class_names = class_names

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """Predict class probabilities for a BGR image array."""
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
        image_batch = np.expand_dims(normalized, axis=0)
        predictions = self.model.predict(image_batch, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        return predictions[0], predicted_class


def load_model_from_zip(zip_bytes: bytes) -> "TeachableMachineModel":
    """Extract keras_model.h5 (+ optional labels.txt) and load into memory.

    Uses a TemporaryDirectory so no temp files leak after the model is loaded.
    """
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)
        extract_dir = os.path.join(tmp, "extracted")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        model_path: Optional[str] = None
        labels_text: Optional[str] = None
        for root, _dirs, files in os.walk(extract_dir):
            for name in files:
                if name == "keras_model.h5":
                    model_path = os.path.join(root, name)
                elif name == "labels.txt":
                    with open(os.path.join(root, name), "r", encoding="utf-8") as lf:
                        labels_text = lf.read()

        if not model_path:
            raise FileNotFoundError("keras_model.h5 not found in zip file")
        model = tf.keras.models.load_model(model_path, compile=False)

    class_names = parse_class_names(labels_text) if labels_text else None
    return TeachableMachineModel(model, class_names)


@st.cache_resource(
    max_entries=config.MODEL_CACHE_MAX_ENTRIES,
    ttl=config.MODEL_CACHE_TTL,
    show_spinner=False,
)
def load_model_cached(file_hash: str, _zip_bytes: bytes) -> "TeachableMachineModel":
    """Cached model load. Keyed only on file_hash (_zip_bytes is not hashed)."""
    return load_model_from_zip(_zip_bytes)


def hash_bytes(data: bytes) -> str:
    """Return the sha256 hex digest of the given bytes."""
    return hashlib.sha256(data).hexdigest()
