"""Tests for config constants."""
import config


def test_config_constants_present_and_typed():
    assert config.IMG_SIZE == 224
    assert config.M_STEPS == 32
    assert config.IG_BATCH == 8
    assert config.TF_INTRA_THREADS == 2
    assert config.TF_INTER_THREADS == 1
    assert config.XAI_SEMAPHORE == 2
    assert config.MODEL_CACHE_MAX_ENTRIES == 8
    assert config.MODEL_CACHE_TTL == 1800
    for name in (
        "IMG_SIZE", "M_STEPS", "IG_BATCH", "TF_INTRA_THREADS",
        "TF_INTER_THREADS", "XAI_SEMAPHORE", "MODEL_CACHE_MAX_ENTRIES",
        "MODEL_CACHE_TTL",
    ):
        assert isinstance(getattr(config, name), int)


def test_rise_and_method_constants():
    assert config.DEFAULT_XAI_METHOD == "Integrated Gradients"
    assert config.RISE_N == 500
    assert config.RISE_GRID == 7
    assert config.RISE_PROB == 0.5
    assert config.RISE_BATCH == 64
    assert config.RISE_SEED == 42
    assert isinstance(config.DEFAULT_XAI_METHOD, str)
    assert isinstance(config.RISE_PROB, float)
    for name in ("RISE_N", "RISE_GRID", "RISE_BATCH", "RISE_SEED"):
        assert isinstance(getattr(config, name), int)
