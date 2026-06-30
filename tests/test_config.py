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
