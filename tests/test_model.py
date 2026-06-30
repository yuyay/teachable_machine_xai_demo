"""Tests for model loading and label parsing."""
import io
import os
import zipfile

import numpy as np
import pytest

import config
from model import parse_class_names, load_model_from_zip, TeachableMachineModel


def test_parse_class_names_strips_index_prefix():
    text = "0 iPhone\n1 Galaxy Phone\n"
    assert parse_class_names(text) == ["iPhone", "Galaxy Phone"]


def test_parse_class_names_handles_single_token_and_blanks():
    text = "Cat\n\n  \nDog\n"
    assert parse_class_names(text) == ["Cat", "Dog"]


def test_load_model_from_zip_returns_model_and_cleans_temp(model_zip_bytes, repo_root):
    before = set(os.listdir(repo_root))
    tm = load_model_from_zip(model_zip_bytes)
    after = set(os.listdir(repo_root))

    assert isinstance(tm, TeachableMachineModel)
    assert tm.class_names == ["ClassA", "ClassB"]
    # input is (None, 224, 224, 3)
    assert tuple(tm.model.input_shape[1:3]) == (config.IMG_SIZE, config.IMG_SIZE)
    # no leftover temp_* files in repo root
    leaked = [f for f in (after - before) if f.startswith("temp_")]
    assert leaked == []


def test_predict_returns_probs_and_index(model_zip_bytes):
    tm = load_model_from_zip(model_zip_bytes)
    image_bgr = (np.random.RandomState(2).rand(240, 320, 3) * 255).astype(np.uint8)
    probs, idx = tm.predict(image_bgr)
    assert probs.ndim == 1
    assert probs.shape[0] >= 2
    assert 0 <= idx < probs.shape[0]
    assert np.all(np.isfinite(probs))


def test_load_model_from_zip_raises_on_missing_model():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("labels.txt", "0 Cat\n")
    with pytest.raises(FileNotFoundError, match="keras_model.h5"):
        load_model_from_zip(buf.getvalue())
