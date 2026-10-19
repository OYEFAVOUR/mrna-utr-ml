"""Tests for loading and validation. No network."""

from __future__ import annotations

import pandas as pd
import pytest

from mrna_utr_ml.data.download import SubsetMode
from mrna_utr_ml.data.load import DataValidationError, load_mpra, validate_mpra


def test_round_trips_synthetic_data(synthetic_csv, synthetic):
    loaded = load_mpra(synthetic_csv)
    assert len(loaded) == len(synthetic)
    assert loaded["utr"].tolist() == synthetic["utr"].tolist()


def test_drops_unused_columns(tmp_path, synthetic):
    """The real file has 32 columns; we must keep only the few we model on."""
    wide = synthetic.copy()
    for i in range(14):
        wide[f"r{i}"] = 0.0
    path = tmp_path / "wide.csv"
    wide.to_csv(path, index=False)
    loaded = load_mpra(path)
    assert set(loaded.columns) == {"utr", "total_reads", "rl", "true_mrl"}


def test_subset_head_takes_deepest_rows(synthetic_csv):
    loaded = load_mpra(synthetic_csv, subset=50, subset_mode=SubsetMode.head)
    full = load_mpra(synthetic_csv)
    assert len(loaded) == 50
    # Because the file is read-depth sorted, head() is the deepest slice.
    assert loaded["total_reads"].min() >= full["total_reads"].median()


def test_subset_random_is_representative_and_seeded(synthetic_csv):
    a = load_mpra(synthetic_csv, subset=200, subset_mode=SubsetMode.random, seed=3)
    b = load_mpra(synthetic_csv, subset=200, subset_mode=SubsetMode.random, seed=3)
    c = load_mpra(synthetic_csv, subset=200, subset_mode=SubsetMode.random, seed=4)
    assert a["utr"].tolist() == b["utr"].tolist()
    assert a["utr"].tolist() != c["utr"].tolist()
    # A random sample should reach shallower rows than head() does.
    head = load_mpra(synthetic_csv, subset=200, subset_mode=SubsetMode.head)
    assert a["total_reads"].min() < head["total_reads"].min()


def test_subset_larger_than_table_is_a_no_op(synthetic_csv):
    assert len(load_mpra(synthetic_csv, subset=10**9)) == 2_000


class TestValidation:
    def test_rejects_missing_column(self):
        with pytest.raises(DataValidationError, match="missing required column"):
            validate_mpra(pd.DataFrame({"utr": ["A" * 50], "rl": [3.0]}))

    def test_rejects_wrong_length_utr(self):
        frame = pd.DataFrame({"utr": ["A" * 49], "total_reads": [10], "rl": [3.0]})
        with pytest.raises(DataValidationError, match="50 nt"):
            validate_mpra(frame)

    def test_rejects_lowercase_sequence(self):
        """Lowercase would silently break uAUG counting via str.find('ATG')."""
        frame = pd.DataFrame({"utr": ["a" * 50], "total_reads": [10], "rl": [3.0]})
        with pytest.raises(DataValidationError, match="unexpected characters"):
            validate_mpra(frame)

    def test_rejects_ambiguous_base(self):
        frame = pd.DataFrame({"utr": ["N" + "A" * 49], "total_reads": [10], "rl": [3.0]})
        with pytest.raises(DataValidationError, match="unexpected characters"):
            validate_mpra(frame)

    def test_rejects_missing_label(self):
        frame = pd.DataFrame({"utr": ["A" * 50], "total_reads": [10], "rl": [None]})
        with pytest.raises(DataValidationError, match="missing values"):
            validate_mpra(frame)

    def test_rejects_non_positive_reads(self):
        frame = pd.DataFrame({"utr": ["A" * 50], "total_reads": [0], "rl": [3.0]})
        with pytest.raises(DataValidationError, match="non-positive"):
            validate_mpra(frame)

    def test_rejects_empty_table(self):
        frame = pd.DataFrame({"utr": [], "total_reads": [], "rl": []})
        with pytest.raises(DataValidationError, match="empty"):
            validate_mpra(frame)

    def test_accepts_valid_table(self, synthetic):
        validate_mpra(synthetic)  # must not raise
