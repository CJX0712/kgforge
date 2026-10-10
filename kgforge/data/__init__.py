"""Data layer: synthetic KG generation + file IO.

Author: 晨星
"""

from .synthetic import (
    generate_synthetic_kg,
    load_kg_from_tsv,
    save_kg_to_tsv,
)

__all__ = ["generate_synthetic_kg", "load_kg_from_tsv", "save_kg_to_tsv"]
