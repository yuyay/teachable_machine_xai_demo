"""Shared pytest fixtures. Placed at repo root so top-level modules import."""
import io
import os
import zipfile

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="session")
def repo_root() -> str:
    return REPO_ROOT


@pytest.fixture(scope="session")
def model_zip_bytes() -> bytes:
    """Build an in-memory Teachable-Machine-style zip from temp_model.h5."""
    h5_path = os.path.join(REPO_ROOT, "temp_model.h5")
    with open(h5_path, "rb") as f:
        h5 = f.read()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("keras_model.h5", h5)
        zf.writestr("labels.txt", "0 ClassA\n1 ClassB\n")
    return buf.getvalue()


@pytest.fixture(scope="session")
def tiny_model():
    """A small deterministic Keras model for fast gradient tests."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(0)
    inp = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(2, 3, padding="same")(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(3, activation="softmax")(x)
    return tf.keras.Model(inp, out)


@pytest.fixture()
def sample_image() -> np.ndarray:
    rng = np.random.RandomState(0)
    return (rng.rand(224, 224, 3).astype(np.float32) * 2.0) - 1.0
