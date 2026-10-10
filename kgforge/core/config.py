"""Configuration with ENV_XXX_* overrides and schema validation.

All tunables live here so the whole system is reproducible from a single object.
Environment overrides use the ``KG_`` prefix, e.g. ``KG_DIM=64 KG_EPOCHS=100``.

Author: 晨星
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from .errors import ConfigError

# ---- schema: name -> (type, default, validator-or-None) ---------------------
_SCHEMA: dict[str, tuple] = {
    "dim": (int, 32, lambda v: 1 < v <= 512),
    "epochs": (int, 400, lambda v: 0 < v <= 5000),
    "batch_size": (int, 64, lambda v: 0 < v <= 8192),
    "learning_rate": (float, 0.05, lambda v: 0.0 < v < 1.0),
    "margin": (float, 3.0, lambda v: 0.0 < v < 20.0),
    "reg": (float, 1e-4, lambda v: 0.0 <= v < 1.0),
    "grad_clip": (float, 1.0, lambda v: 0.0 <= v <= 100.0),
    "negatives": (int, 1, lambda v: 0 < v <= 32),
    "seed": (int, 7, lambda v: 0 <= v < 2**31),
    "n_entities": (int, 300, lambda v: 10 <= v <= 200000),
    "n_relations": (int, 8, lambda v: 1 <= v <= 5000),
    "n_train": (int, 1000, lambda v: 50 <= v <= 500000),
    "n_valid": (int, 200, lambda v: 10 <= v <= 50000),
    "n_test": (int, 200, lambda v: 10 <= v <= 50000),
    "difficulty": (str, "medium", lambda v: v in ("easy", "medium", "hard")),
    "top_k": (tuple, (1, 3, 10), None),
    "use_complex": (bool, True, None),
    "use_rotate": (bool, True, None),
}


@dataclass
class Config:
    dim: int = 32
    epochs: int = 400
    batch_size: int = 64
    learning_rate: float = 0.05
    margin: float = 3.0
    reg: float = 1e-4
    grad_clip: float = 1.0
    negatives: int = 1
    seed: int = 7
    n_entities: int = 300
    n_relations: int = 8
    n_train: int = 1000
    n_valid: int = 200
    n_test: int = 200
    difficulty: str = "medium"
    top_k: tuple = field(default_factory=lambda: (1, 3, 10))
    use_complex: bool = True
    use_rotate: bool = True

    # ----------------------------- overrides ---------------------------------
    @classmethod
    def from_env(cls, prefix: str = "KG_") -> Config:
        """Build a Config, applying ``KG_<NAME>`` environment overrides."""
        # PEP 563 safe defaults already set; read overrides.
        overrides: dict[str, Any] = {}
        for name, (typ, _, _) in _SCHEMA.items():
            env_key = prefix + name.upper()
            if env_key in os.environ:
                raw = os.environ[env_key]
                try:
                    if typ is bool:
                        overrides[name] = raw.lower() in ("1", "true", "yes", "on")
                    elif typ is tuple:
                        overrides[name] = tuple(int(x) for x in raw.split(",") if x != "")
                    else:
                        overrides[name] = typ(raw)
                except (ValueError, TypeError) as exc:
                    raise ConfigError(f"{env_key}={raw!r} not parsable as {typ.__name__}") from exc
        cfg = cls(**overrides)
        cfg.validate()
        return cfg

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Config:
        cfg = cls(**{k: v for k, v in d.items() if k in _SCHEMA})
        cfg.validate()
        return cfg

    # ----------------------------- validation --------------------------------
    def validate(self) -> None:
        data = asdict(self)
        for name, (typ, _, ok) in _SCHEMA.items():
            val = data[name]
            # top_k is a tuple (already handled); bool is a subclass of int
            if not isinstance(val, typ) and not (typ is int and isinstance(val, bool)):
                raise ConfigError(f"{name}={val!r} is not {typ.__name__}")
            if ok is not None and not ok(val):
                raise ConfigError(f"{name}={val!r} violates schema")
        # capacity = distinct (h, r, t) triples the id space can express
        if self.n_train + self.n_valid + self.n_test > (self.n_entities**2) * self.n_relations:
            # soft guard: too many requested facts for the id space
            raise ConfigError("requested (n_train+n_valid+n_test) exceeds id space capacity")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_cli_summary(self) -> str:
        return " ".join(f"{f.name}={getattr(self, f.name)!r}" for f in fields(self))
