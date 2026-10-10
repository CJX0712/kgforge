"""Core layer tests: config schema / ENV overrides / seeding determinism.

Author: 晨星
"""

from __future__ import annotations

import numpy as np
import pytest

from kgforge.core.config import Config
from kgforge.core.errors import ConfigError
from kgforge.core.seed import set_all


def test_default_config_validates():
    Config().validate()


def test_env_override_applied(monkeypatch):
    monkeypatch.setenv("KG_DIM", "16")
    monkeypatch.setenv("KG_EPOCHS", "3")
    monkeypatch.setenv("KG_USE_ROTATE", "false")
    cfg = Config.from_env()
    assert cfg.dim == 16
    assert cfg.epochs == 3
    assert cfg.use_rotate is False


def test_env_override_bad_value_raises(monkeypatch):
    monkeypatch.setenv("KG_DIM", "not-an-int")
    with pytest.raises(ConfigError):
        Config.from_env()


def test_schema_violation_raises():
    with pytest.raises(ConfigError):
        Config(epochs=-1).validate()
    with pytest.raises(ConfigError):
        Config(difficulty="impossible").validate()


def test_seed_is_reproducible():
    rng1 = set_all(1234)
    rng2 = set_all(1234)
    a = rng1.standard_normal(5)
    b = rng2.standard_normal(5)
    assert np.array_equal(a, b)


def test_seed_different_gives_different_stream():
    rng1 = set_all(1)
    rng2 = set_all(2)
    assert not np.array_equal(rng1.standard_normal(5), rng2.standard_normal(5))


def test_config_roundtrip_dict():
    cfg = Config(dim=12, epochs=7)
    assert Config.from_dict(cfg.as_dict()).dim == 12
    assert Config.from_dict(cfg.as_dict()).epochs == 7
