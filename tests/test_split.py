"""Tests for splitting.

The grouping machinery is hard to trust on this dataset, because the real
library is random 50-mers where near-duplicates essentially do not occur -- so
a test that merely runs it on random data would pass even if it were broken.
These tests therefore **plant** near-duplicates and require them to be found,
and plant unrelated sequences and require them not to be merged.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mrna_utr_ml import UTR_LENGTH
from mrna_utr_ml.data.split import (
    SplitStrategy,
    assert_no_leakage,
    collapse_exact_duplicates,
    describe_shift,
    group_near_duplicates,
    kmer_id_matrix,
    make_splits,
    minhash_signatures,
    write_manifest,
)
from mrna_utr_ml.data.synthetic import make_synthetic_mpra


def mutate_at(seq: str, positions, rng) -> str:
    """Change exactly `positions` of `seq`, each to a different base."""
    chars = list(seq)
    for pos in positions:
        chars[pos] = rng.choice([b for b in "ACGT" if b != chars[pos]])
    return "".join(chars)


def mutate(seq: str, n: int, rng) -> str:
    """Change `n` randomly chosen positions of `seq` to a different base."""
    return mutate_at(seq, rng.choice(len(seq), n, replace=False), rng)


class TestKmerEncoding:
    def test_shape_and_range(self):
        ids = kmer_id_matrix(["ACGT" * 12 + "AC"], k=6)
        assert ids.shape == (1, UTR_LENGTH - 6 + 1)
        assert ids.min() >= 0 and ids.max() < 4**6

    def test_encoding_is_positional(self):
        # AAAAAA -> 0; AAAAAC -> 1; CAAAAA -> 4**5
        ids = kmer_id_matrix(["A" * 50, "A" * 49 + "C", "C" + "A" * 49], k=6)
        assert ids[0, 0] == 0
        assert ids[1, -1] == 1
        assert ids[2, 0] == 4**5

    def test_rejects_ragged_input(self):
        with pytest.raises(ValueError, match="equal-length"):
            kmer_id_matrix(["ACGTACGT", "ACGT"], k=4)

    def test_rejects_non_acgt(self):
        with pytest.raises(ValueError, match="only A, C, G, T"):
            kmer_id_matrix(["N" * 50], k=6)


class TestMinHash:
    def test_signature_similarity_tracks_jaccard(self):
        """The property the whole LSH stage rests on.

        The fraction of signature positions where two sequences agree should
        approximate the Jaccard similarity of their k-mer sets. If this drifts,
        the candidate search silently stops finding real near-duplicates.
        """
        rng = np.random.default_rng(0)
        base = "".join(rng.choice(list("ACGT"), UTR_LENGTH))
        near = mutate(base, 2, rng)
        ids = kmer_id_matrix([base, near], k=6)
        sets = [set(row.tolist()) for row in ids]
        true_jaccard = len(sets[0] & sets[1]) / len(sets[0] | sets[1])

        sig = minhash_signatures(ids, n_perm=512, seed=0)
        estimated = float((sig[0] == sig[1]).mean())
        assert estimated == pytest.approx(true_jaccard, abs=0.08)

    def test_identical_sequences_have_identical_signatures(self):
        ids = kmer_id_matrix(["ACGT" * 12 + "AC"] * 2, k=6)
        sig = minhash_signatures(ids, n_perm=64, seed=0)
        assert np.array_equal(sig[0], sig[1])

    def test_is_deterministic_for_a_seed(self):
        ids = kmer_id_matrix(["ACGT" * 12 + "AC"], k=6)
        assert np.array_equal(
            minhash_signatures(ids, n_perm=64, seed=3),
            minhash_signatures(ids, n_perm=64, seed=3),
        )


class TestGrouping:
    def test_finds_planted_near_duplicates(self):
        """A planted family of near-identical UTRs must end up in one group."""
        rng = np.random.default_rng(1)
        base = "".join(rng.choice(list("ACGT"), UTR_LENGTH))
        family = [base] + [mutate(base, 1, rng) for _ in range(4)]
        unrelated = ["".join(rng.choice(list("ACGT"), UTR_LENGTH)) for _ in range(40)]

        groups = group_near_duplicates(family + unrelated, threshold=0.5, seed=0)
        assert len(set(groups[:5])) == 1, "planted family was split across groups"

    def test_does_not_merge_unrelated_random_sequences(self):
        """The false-positive side. Random 50-mers share almost no 6-mers."""
        rng = np.random.default_rng(2)
        seqs = ["".join(rng.choice(list("ACGT"), UTR_LENGTH)) for _ in range(300)]
        groups = group_near_duplicates(seqs, threshold=0.5, seed=0)
        assert len(set(groups)) == len(seqs), "unrelated sequences were merged"

    def test_grouping_is_transitive(self):
        """A~B and B~C must put A, B and C together even if A and C are not similar.

        This is why union-find is used rather than pairwise assignment. Without
        it, A and C could land either side of a split while both sit next to B.

        Mutation sites are drawn **disjointly** so the second step cannot revert
        the first; with overlapping sites A and C come out more similar than
        intended and the test silently stops exercising transitivity. The
        premise is asserted rather than assumed, and the threshold is placed
        between the one-step and two-step similarities that were actually
        measured.
        """
        rng = np.random.default_rng(3)
        a = "".join(rng.choice(list("ACGT"), UTR_LENGTH))
        sites = rng.permutation(UTR_LENGTH)
        b = mutate_at(a, sites[:4], rng)
        c = mutate_at(b, sites[4:8], rng)

        sets = [set(row.tolist()) for row in kmer_id_matrix([a, b, c], k=6)]

        def jaccard(i, j):
            return len(sets[i] & sets[j]) / len(sets[i] | sets[j])

        one_step = min(jaccard(0, 1), jaccard(1, 2))
        two_step = jaccard(0, 2)
        assert two_step < one_step, (
            f"premise failed: A~C ({two_step:.3f}) is not below the one-step "
            f"similarity ({one_step:.3f}), so this would not test transitivity"
        )
        threshold = (one_step + two_step) / 2

        groups = group_near_duplicates([a, b, c], threshold=threshold, seed=0)
        assert len(set(groups)) == 1

    def test_handles_trivial_inputs(self):
        assert len(group_near_duplicates([])) == 0
        assert len(group_near_duplicates(["A" * UTR_LENGTH])) == 1

    def test_is_deterministic(self):
        rng = np.random.default_rng(4)
        seqs = ["".join(rng.choice(list("ACGT"), UTR_LENGTH)) for _ in range(60)]
        assert np.array_equal(
            group_near_duplicates(seqs, seed=7), group_near_duplicates(seqs, seed=7)
        )


class TestCollapseExactDuplicates:
    def test_no_duplicates_is_a_no_op(self, synthetic):
        collapsed, n = collapse_exact_duplicates(synthetic)
        assert n == 0
        assert len(collapsed) == len(synthetic)

    def test_merges_and_sums_reads(self):
        frame = pd.DataFrame(
            {
                "utr": ["A" * 50, "A" * 50, "C" * 50],
                "total_reads": [100, 300, 50],
                "rl": [4.0, 8.0, 5.0],
            }
        )
        collapsed, n = collapse_exact_duplicates(frame)
        assert n == 1
        assert len(collapsed) == 2
        row = collapsed[collapsed["utr"] == "A" * 50].iloc[0]
        assert row["total_reads"] == 400

    def test_mrl_is_read_weighted_not_a_plain_mean(self):
        """A 300-read estimate must outweigh a 100-read one.

        Plain mean would give 6.0; read-weighted gives 7.0. Using the plain mean
        would quietly discard the fact that one estimate is three times better
        measured than the other.
        """
        frame = pd.DataFrame(
            {"utr": ["A" * 50] * 2, "total_reads": [100, 300], "rl": [4.0, 8.0]}
        )
        collapsed, _ = collapse_exact_duplicates(frame)
        assert collapsed["rl"].iloc[0] == pytest.approx(7.0)


class TestMakeSplits:
    @pytest.fixture(scope="class")
    @classmethod
    def data(cls):
        return make_synthetic_mpra(3_000, seed=0)

    def test_splits_are_disjoint_and_cover_everything(self, data):
        frame, split = make_splits(data, test_size=400, val_size=400, seed=0)
        all_idx = np.concatenate([split.train, split.val, split.test])
        assert len(all_idx) == len(frame)
        assert len(set(all_idx.tolist())) == len(frame)

    def test_no_sequence_leaks_between_splits(self, data):
        frame, split = make_splits(data, test_size=400, val_size=400, seed=0)
        assert_no_leakage(frame, split)  # must not raise

    def test_published_strategy_takes_the_deepest_reads(self, data):
        frame, split = make_splits(
            data, strategy=SplitStrategy.published, test_size=400, val_size=400, seed=0
        )
        test_reads = frame["total_reads"].iloc[split.test]
        train_reads = frame["total_reads"].iloc[split.train]
        assert test_reads.min() >= train_reads.max()

    def test_random_strategy_does_not_select_on_depth(self, data):
        """The point of the random split: its test set must not be depth-shifted."""
        frame, split = make_splits(
            data, strategy=SplitStrategy.random, test_size=400, val_size=400, seed=0
        )
        test_median = frame["total_reads"].iloc[split.test].median()
        train_median = frame["total_reads"].iloc[split.train].median()
        assert abs(test_median - train_median) / train_median < 0.25

    def test_sizes_are_close_to_requested(self, data):
        """Whole groups only, so sizes land near the target rather than on it."""
        frame, split = make_splits(data, test_size=400, val_size=400, seed=0)
        assert 400 <= len(split.test) < 450
        assert 400 <= len(split.val) < 450

    def test_is_reproducible_for_a_seed(self, data):
        a = make_splits(data, test_size=300, val_size=300, seed=11)[1]
        b = make_splits(data, test_size=300, val_size=300, seed=11)[1]
        assert np.array_equal(a.test, b.test)
        assert np.array_equal(a.train, b.train)

    def test_different_seeds_give_different_splits(self, data):
        a = make_splits(data, test_size=300, val_size=300, seed=1)[1]
        b = make_splits(data, test_size=300, val_size=300, seed=2)[1]
        assert not np.array_equal(a.test, b.test)

    def test_planted_duplicates_never_straddle_a_split(self):
        """The leakage test that actually bites.

        A family of near-identical UTRs is planted, then split. Every member
        must land on the same side. Without grouping this fails roughly whenever
        the family is large enough to be sampled into two splits.
        """
        rng = np.random.default_rng(5)
        base_frame = make_synthetic_mpra(600, seed=0)
        base = base_frame["utr"].iloc[0]
        family = [mutate(base, 1, rng) for _ in range(30)]
        planted = pd.DataFrame(
            {
                "utr": family,
                "total_reads": rng.integers(100, 900, len(family)),
                "rl": rng.normal(6, 1, len(family)),
                "true_mrl": rng.normal(6, 1, len(family)),
            }
        )
        data = pd.concat([base_frame, planted], ignore_index=True)

        frame, split = make_splits(
            data, test_size=120, val_size=120, seed=0, jaccard_threshold=0.5
        )
        where = {}
        for name, idx in (("train", split.train), ("val", split.val), ("test", split.test)):
            for seq in frame["utr"].iloc[idx]:
                if seq in set(family) or seq == base:
                    where[seq] = name
        assert len(set(where.values())) == 1, f"planted family straddled splits: {where}"

    def test_rejects_impossible_sizes(self, data):
        with pytest.raises(ValueError, match="smaller than the dataset"):
            make_splits(data, test_size=2_000, val_size=2_000)

    def test_leakage_assertion_catches_a_bad_split(self, data):
        """Prove the guard fails when it should, not just that it passes."""
        frame, split = make_splits(data, test_size=300, val_size=300, seed=0)
        broken = type(split)(
            train=np.concatenate([split.train, split.test[:5]]),
            val=split.val,
            test=split.test,
            strategy=split.strategy,
            seed=split.seed,
            n_groups=split.n_groups,
            n_grouped_sequences=split.n_grouped_sequences,
            n_exact_duplicates_collapsed=split.n_exact_duplicates_collapsed,
            jaccard_threshold=split.jaccard_threshold,
        )
        with pytest.raises(AssertionError, match="appear in both"):
            assert_no_leakage(frame, broken)


class TestReporting:
    def test_describe_shift_reports_every_split(self, synthetic):
        frame, split = make_splits(synthetic, test_size=200, val_size=200, seed=0)
        table = describe_shift(frame, split)
        assert list(table["split"]) == ["train", "val", "test"]
        assert table["n"].sum() == len(frame)
        assert (table["uaug_per_utr"] >= 0).all()

    def test_published_split_shows_the_depth_shift(self, synthetic):
        """describe_shift must actually surface the shift it exists to surface."""
        frame, split = make_splits(
            synthetic, strategy=SplitStrategy.published, test_size=200, val_size=200
        )
        table = describe_shift(frame, split).set_index("split")
        assert table.loc["test", "median_reads"] > table.loc["train", "median_reads"]

    def test_manifest_round_trips(self, synthetic, tmp_path):
        frame, split = make_splits(synthetic, test_size=200, val_size=200, seed=0)
        path = tmp_path / "split_manifest.json"
        manifest = write_manifest(frame, split, path)
        loaded = json.loads(path.read_text())
        assert loaded == manifest
        assert set(loaded["digests"]) == {"train", "val", "test"}
        assert len(loaded["digests"]["test"]) == 64

    def test_manifest_digest_detects_a_changed_split(self, synthetic, tmp_path):
        a = write_manifest(*make_splits(synthetic, test_size=200, val_size=200, seed=1),
                           tmp_path / "a.json")
        b = write_manifest(*make_splits(synthetic, test_size=200, val_size=200, seed=2),
                           tmp_path / "b.json")
        assert a["digests"]["test"] != b["digests"]["test"]

    def test_manifest_digest_is_order_independent(self, synthetic, tmp_path):
        """Same set of sequences, different order, same digest."""
        frame, split = make_splits(synthetic, test_size=200, val_size=200, seed=0)
        shuffled = type(split)(
            train=split.train,
            val=split.val,
            test=split.test[::-1].copy(),
            strategy=split.strategy,
            seed=split.seed,
            n_groups=split.n_groups,
            n_grouped_sequences=split.n_grouped_sequences,
            n_exact_duplicates_collapsed=split.n_exact_duplicates_collapsed,
            jaccard_threshold=split.jaccard_threshold,
        )
        a = write_manifest(frame, split, tmp_path / "a.json")
        b = write_manifest(frame, shuffled, tmp_path / "b.json")
        assert a["digests"]["test"] == b["digests"]["test"]


class TestFivePrimeArtefact:
    """Pin the near-duplicate artefact found in the real library.

    Grouping was expected to be a formality on a random 50-mer library. It is
    not: 6,324 groups covering 3.9% of the real data, overwhelmingly pairs
    differing by one base at position 0 or 1. These tests assert that the
    detector would still catch that pattern, so a future change to k, the
    threshold or the banding cannot silently stop finding it.
    """

    @staticmethod
    def _one_base_variant(seq: str, position: int) -> str:
        replacement = "A" if seq[position] != "A" else "C"
        return seq[:position] + replacement + seq[position + 1 :]

    @pytest.mark.parametrize("position", [0, 1, 25, 49])
    def test_single_base_variants_are_grouped_anywhere_in_the_sequence(self, position):
        rng = np.random.default_rng(10)
        base = "".join(rng.choice(list("ACGT"), UTR_LENGTH))
        others = ["".join(rng.choice(list("ACGT"), UTR_LENGTH)) for _ in range(100)]
        variant = self._one_base_variant(base, position)
        groups = group_near_duplicates([base, variant] + others, seed=0)
        assert groups[0] == groups[1], (
            f"a one-base variant at position {position} was not grouped with its parent"
        )

    def test_terminal_variants_are_not_a_blind_spot(self):
        """Position 0 and 49 change the fewest k-mers, so they are the hardest
        case for a k-mer detector -- and position 0 is exactly where the real
        artefact lives. A 6-mer at the very end appears in only one k-mer
        window, so the similarity drop is smallest there."""
        rng = np.random.default_rng(11)
        base = "".join(rng.choice(list("ACGT"), UTR_LENGTH))
        sets = [
            set(row.tolist())
            for row in kmer_id_matrix(
                [base, self._one_base_variant(base, 0), self._one_base_variant(base, 25)],
                k=6,
            )
        ]
        edge = len(sets[0] & sets[1]) / len(sets[0] | sets[1])
        middle = len(sets[0] & sets[2]) / len(sets[0] | sets[2])
        # A terminal change is *less* disruptive, so it must clear the threshold
        # comfortably; this documents why the default threshold of 0.6 works.
        assert edge > middle > 0.6

    def test_grouped_variants_never_straddle_a_split(self):
        """The end-to-end guarantee: a one-base variant of a training sequence
        must never appear in the test set."""
        rng = np.random.default_rng(12)
        base_frame = make_synthetic_mpra(800, seed=0)
        parents = base_frame["utr"].iloc[:40].tolist()
        variants = [self._one_base_variant(s, 0) for s in parents]
        planted = pd.DataFrame(
            {
                "utr": variants,
                "total_reads": rng.integers(20, 120, len(variants)),  # shallow, as in reality
                "rl": rng.normal(6, 2, len(variants)),
                "true_mrl": rng.normal(6, 2, len(variants)),
            }
        )
        data = pd.concat([base_frame, planted], ignore_index=True)
        frame, split = make_splits(data, test_size=150, val_size=150, seed=0)

        location = {}
        for name, idx in (("train", split.train), ("val", split.val), ("test", split.test)):
            for seq in frame["utr"].iloc[idx]:
                location[seq] = name
        for parent, variant in zip(parents, variants, strict=True):
            if parent in location and variant in location:
                assert location[parent] == location[variant], (
                    f"parent in {location[parent]} but its one-base variant in "
                    f"{location[variant]}"
                )
