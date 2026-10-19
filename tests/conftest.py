"""Shared fixtures. Everything here is offline and CPU-only."""

from __future__ import annotations

import pandas as pd
import pytest

from mrna_utr_ml.data.synthetic import make_synthetic_mpra


@pytest.fixture(scope="session")
def synthetic() -> pd.DataFrame:
    """A modest synthetic dataset, generated once per test session."""
    return make_synthetic_mpra(2_000, seed=0)


@pytest.fixture
def synthetic_csv(tmp_path, synthetic):
    """The synthetic dataset written to disk, for exercising the loader."""
    path = tmp_path / "synthetic.csv"
    synthetic.to_csv(path, index=False)
    return path
