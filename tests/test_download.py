"""Tests for the download helpers that do not require network access.

The actual GEO transfer is not tested here on purpose: CI must stay offline and
fast. What is tested is the integrity machinery around it, which is where the
real risk of silent corruption lies.
"""

from __future__ import annotations

import hashlib

from mrna_utr_ml.data.download import (
    EXPECTED_BYTES,
    GEO_URL,
    SubsetMode,
    record_checksum,
    sha256_file,
)


def test_geo_url_points_at_the_expected_sample():
    assert GEO_URL.startswith("https://")
    assert "GSM3130435_egfp_unmod_1.csv.gz" in GEO_URL
    assert EXPECTED_BYTES > 60_000_000


def test_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "f.bin"
    payload = b"ACGT" * 10_000
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_sha256_is_chunk_size_independent(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"x" * 5_000)
    assert sha256_file(path, chunk_bytes=7) == sha256_file(path, chunk_bytes=4096)


def test_record_checksum_creates_sha256sum_format(tmp_path):
    target = tmp_path / "data.csv.gz"
    target.write_bytes(b"hello")
    checksums = tmp_path / "CHECKSUMS.txt"
    record_checksum(target, sha256_file(target), checksums)
    line = checksums.read_text().strip()
    value, name = line.split(maxsplit=1)
    assert name == "data.csv.gz"
    assert value == hashlib.sha256(b"hello").hexdigest()


def test_record_checksum_updates_rather_than_duplicates(tmp_path):
    target = tmp_path / "data.csv.gz"
    target.write_bytes(b"v1")
    checksums = tmp_path / "CHECKSUMS.txt"
    record_checksum(target, "aaa", checksums)
    record_checksum(target, "bbb", checksums)
    lines = [ln for ln in checksums.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    assert lines[0].startswith("bbb")


def test_record_checksum_keeps_other_entries(tmp_path):
    checksums = tmp_path / "CHECKSUMS.txt"
    (tmp_path / "a").write_bytes(b"a")
    (tmp_path / "b").write_bytes(b"b")
    record_checksum(tmp_path / "a", "aaa", checksums)
    record_checksum(tmp_path / "b", "bbb", checksums)
    text = checksums.read_text()
    assert "a\n" in text and "b\n" in text


def test_subset_mode_values_are_cli_friendly():
    assert SubsetMode.head == "head"
    assert SubsetMode.random == "random"
