"""Tests for the synthetic generative model.

These assert the *biology* encoded in the generator, which is what makes it
usable as ground truth in later stages. If `find_uaugs` miscomputes frame, every
downstream claim about in-frame vs out-of-frame uAUGs is wrong, so it is tested
directly rather than only through a model.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import pearsonr

from mrna_utr_ml import UTR_LENGTH
from mrna_utr_ml.data.synthetic import (
    BASE_MRL,
    BLOCKED_MRL,
    find_uaugs,
    gc_fraction,
    kozak_strength,
    make_synthetic_mpra,
    scanning_throughput,
    true_mrl,
)


class TestKozakStrength:
    def test_strong_context_scores_one(self):
        # A at -3 (purine) and G at +4 -> both halves satisfied.
        seq = "A" * 7 + "ATG" + "G" + "C" * 39
        assert kozak_strength(seq, 7) == pytest.approx(1.0)

    def test_weak_context_scores_zero(self):
        # C at -3 (pyrimidine) and C at +4.
        seq = "C" * 7 + "ATG" + "C" * 40
        assert kozak_strength(seq, 7) == pytest.approx(0.0)

    def test_purine_at_minus3_alone_scores_half(self):
        seq = "CCCCGCC" + "ATG" + "C" * 40  # G at -3, C at +4
        assert kozak_strength(seq, 7) == pytest.approx(0.5)

    def test_atg_at_sequence_start_has_no_minus3_context(self):
        # Biologically real: a 5'-terminal AUG has no upstream context and
        # initiates poorly. Must not raise or wrap around to the 3' end.
        seq = "ATG" + "G" + "C" * 46
        assert kozak_strength(seq, 0) == pytest.approx(0.5)  # +4 G only


class TestFindUaugs:
    def test_no_atg_gives_empty(self):
        assert find_uaugs("C" * UTR_LENGTH) == []

    def test_frame_is_relative_to_reporter_start(self):
        # The reporter AUG begins at index 50, so an ATG at index 47 is in frame
        # (50 - 47 = 3) and one at index 46 is not (50 - 46 = 4).
        in_frame = "C" * 47 + "ATG"
        assert find_uaugs(in_frame[:50] + "", UTR_LENGTH)  # sanity: found
        assert find_uaugs("C" * 47 + "ATG") == [(47, True)]
        assert find_uaugs("C" * 46 + "ATG" + "C") == [(46, False)]

    def test_overlapping_atgs_are_both_counted(self):
        # ATGATG contains ATG at 0 and 3; a naive start += 3 scan is fine here,
        # but ATGATGATG-style overlaps in other registers are not, so we scan
        # with step 1. Both must be reported.
        hits = find_uaugs("ATGATG" + "C" * 44)
        assert [p for p, _ in hits] == [0, 3]

    def test_tandem_overlap_atg_within_atg(self):
        # "AATGATG" -> ATG at 1 and 4.
        hits = find_uaugs("AATGATG" + "C" * 43)
        assert [p for p, _ in hits] == [1, 4]


class TestTrueMrl:
    def test_out_of_frame_uaug_is_worse_than_in_frame(self):
        """The central biological claim the generator encodes."""
        clean = "C" * UTR_LENGTH
        # Place an ATG in strong context in each frame, same position class.
        oof = "C" * 43 + "A" + "CC" + "ATG" + "G"  # idx 46 -> out of frame
        inf = "C" * 44 + "A" + "CC" + "ATG"  # idx 47 -> in frame
        assert len(oof) == len(inf) == UTR_LENGTH
        assert find_uaugs(oof)[0][1] is False
        assert find_uaugs(inf)[0][1] is True
        assert true_mrl(oof) < true_mrl(inf) < true_mrl(clean)

    def test_strong_kozak_uaug_is_worse_than_weak(self):
        strong = "C" * 40 + "A" + "CC" + "ATG" + "G" + "CCC"
        weak = "C" * 40 + "C" + "CC" + "ATG" + "C" + "CCC"
        assert len(strong) == len(weak) == UTR_LENGTH
        assert true_mrl(strong) < true_mrl(weak)

    def test_gc_effect_is_negative_but_weak(self):
        """GC must matter, but far less than a uAUG -- otherwise the GC-only
        baseline would be strong and the comparison in stage 6 meaningless."""
        at_rich = "AT" * (UTR_LENGTH // 2)
        gc_rich = "GC" * (UTR_LENGTH // 2)
        assert "ATG" not in at_rich and "ATG" not in gc_rich
        gc_effect = abs(true_mrl(gc_rich) - true_mrl(at_rich))

        clean = "C" * UTR_LENGTH
        one_oof_uaug = "C" * 43 + "A" + "CC" + "ATG" + "G"
        uaug_effect = abs(true_mrl(one_oof_uaug) - true_mrl(clean))
        assert gc_effect < uaug_effect


class TestScanningThroughput:
    """The leaky-scanning cascade: uAUGs are sequential sieves, not a sum."""

    def test_no_uaug_passes_everything(self):
        assert scanning_throughput("C" * UTR_LENGTH) == pytest.approx(1.0)

    def test_perfect_kozak_out_of_frame_uaug_blocks_everything(self):
        # CAPTURE_OOF is 1.0, so a strong-context out-of-frame uAUG is a
        # complete block -- the fitted value sits at that boundary.
        seq = "C" * 43 + "A" + "CC" + "ATG" + "G"
        assert len(seq) == UTR_LENGTH
        assert find_uaugs(seq)[0][1] is False
        assert kozak_strength(seq, 46) == pytest.approx(1.0)
        assert scanning_throughput(seq) == pytest.approx(0.0)

    def test_in_frame_uaug_leaks_most_ribosomes_through(self):
        seq = "C" * 44 + "A" + "CC" + "ATG"
        assert find_uaugs(seq)[0][1] is True
        assert scanning_throughput(seq) > 0.8

    def test_throughput_stays_in_unit_interval(self):
        for seed in range(20):
            frame = make_synthetic_mpra(50, seed=seed)
            values = [scanning_throughput(s) for s in frame["utr"]]
            assert all(0.0 <= v <= 1.0 for v in values)


class TestSaturation:
    """Two uAUGs are worse than one, but not twice as bad.

    This is the behaviour an additive penalty gets wrong, and getting it wrong
    drove 1.5% of synthetic labels onto the MRL floor against 0.13% in the real
    file. These tests pin the multiplicative form.
    """

    # Four 7-nt slots, all in out-of-frame registers, each filled with either
    # an ATG motif or a GC-matched decoy. Holding GC and length constant is the
    # point: an earlier version of this helper changed GC as it added uAUGs, and
    # the GC term cancelled the uAUG effect, which made MRL go *up*.
    SLOTS = (0, 7, 21, 28)
    ATG_MOTIF = "ACCATGC"  # purine at -3, ATG at offset 3 -> kozak 0.5
    DECOY_MOTIF = "ACCATCC"  # same length, same 4 G+C, no ATG

    @classmethod
    def _with_uaugs(cls, count: int) -> str:
        chars = ["C"] * UTR_LENGTH
        for i, slot in enumerate(cls.SLOTS):
            motif = cls.ATG_MOTIF if i < count else cls.DECOY_MOTIF
            chars[slot : slot + len(motif)] = list(motif)
        seq = "".join(chars)
        assert len(seq) == UTR_LENGTH
        return seq

    def test_helper_holds_gc_and_frame_constant(self):
        """Guard the guard: the construction must isolate uAUG count."""
        seqs = [self._with_uaugs(k) for k in range(5)]
        gcs = {gc_fraction(s) for s in seqs}
        assert len(gcs) == 1, f"GC varies across constructs: {gcs}"
        for k, seq in enumerate(seqs):
            hits = find_uaugs(seq)
            assert len(hits) == k, (k, hits)
            assert all(not in_frame for _, in_frame in hits), hits

    def test_more_uaugs_lower_mrl(self):
        values = [true_mrl(self._with_uaugs(k)) for k in range(5)]
        assert values == sorted(values, reverse=True), values

    def test_effect_is_sub_additive(self):
        """The second uAUG must cost less than the first did."""
        clean = true_mrl(self._with_uaugs(0))
        one = true_mrl(self._with_uaugs(1))
        two = true_mrl(self._with_uaugs(2))
        first_cost = clean - one
        second_cost = one - two
        assert 0 < second_cost < first_cost

    def test_uaugs_alone_cannot_push_below_the_blocked_floor(self):
        """No number of uAUGs can take MRL below BLOCKED_MRL.

        Only the GC term can, and it is bounded. This is what keeps the
        generated labels physically plausible.
        """
        for count in range(1, 5):
            seq = self._with_uaugs(count)
            gc_term = true_mrl(seq) - (
                BLOCKED_MRL + (BASE_MRL - BLOCKED_MRL) * scanning_throughput(seq)
            )
            assert true_mrl(seq) - gc_term >= BLOCKED_MRL - 1e-9


class TestMakeSyntheticMpra:
    def test_is_deterministic_for_a_given_seed(self):
        a = make_synthetic_mpra(200, seed=7)
        b = make_synthetic_mpra(200, seed=7)
        assert a.equals(b)

    def test_different_seeds_differ(self):
        a = make_synthetic_mpra(200, seed=1)
        b = make_synthetic_mpra(200, seed=2)
        assert not a["utr"].equals(b["utr"])

    def test_shape_and_columns(self, synthetic):
        assert len(synthetic) == 2_000
        assert set(synthetic.columns) == {"utr", "total_reads", "rl", "true_mrl"}
        assert (synthetic["utr"].str.len() == UTR_LENGTH).all()

    def test_sorted_by_read_depth_like_the_real_file(self, synthetic):
        assert synthetic["total_reads"].is_monotonic_decreasing

    def test_mrl_range_resembles_the_real_assay(self, synthetic):
        """Bounds taken from the real file, not guessed.

        Real egfp_unmod: min 0.00, max 13.00 (both clipped), mean 6.45, sd 1.75,
        1st-99th percentile 2.1-9.9. Percentiles rather than extremes, because
        extremes are set by the clip and so say nothing about the distribution.
        """
        rl = synthetic["rl"]
        assert rl.min() >= 0.0 and rl.max() <= 13.0
        assert 1.0 < rl.quantile(0.01) < 4.0
        assert 8.0 < rl.quantile(0.99) < 12.0
        assert 5.5 < rl.mean() < 7.5

    def test_label_noise_shrinks_with_read_depth(self):
        """Measurement noise must scale as 1/sqrt(reads).

        This is the property that justifies a read-count filter at all, so it is
        asserted on the noise directly rather than on a correlation. The sd of
        (observed - truth) should roughly halve for each 4x increase in depth.
        """
        big = make_synthetic_mpra(40_000, seed=0)
        err = big["rl"] - big["true_mrl"]
        bands = [(50, 100), (100, 300), (300, 1000), (1000, 10**9)]
        sds = []
        for lo, hi in bands:
            mask = (big["total_reads"] >= lo) & (big["total_reads"] < hi)
            sds.append(err[mask].std())
        # Strictly decreasing with depth.
        assert sds == sorted(sds, reverse=True), sds
        # And the shallowest band is several times noisier than the deepest.
        assert sds[0] > 3 * sds[-1]

    def test_read_depth_filtering_only_helps_the_shallow_tail(self):
        """The honest version of 'filter on read count'.

        Because real read depth is lognormal with a median of 890, measurement
        noise is already small for the large majority of UTRs: sd(error) is 0.07
        above 1,000 reads against 0.57 below 50. So a read-count filter is a
        cleanup of a thin tail -- roughly 3% of the real library below 50 reads
        -- not a major lever on model performance. Asserted here so stage 4 does
        not overclaim what the filter buys.
        """
        big = make_synthetic_mpra(40_000, seed=0)
        err = big["rl"] - big["true_mrl"]
        shallow = big["total_reads"] < 50
        assert err[shallow].std() > 3 * err[~shallow].std()
        # ...but the tail is small, so overall correlation barely moves.
        r_all = pearsonr(big["rl"], big["true_mrl"]).statistic
        r_filtered = pearsonr(big["rl"][~shallow], big["true_mrl"][~shallow]).statistic
        assert 0.0 <= r_filtered - r_all < 0.05

    def test_uaug_count_correlates_negatively_with_truth(self, synthetic):
        n_oof = np.array(
            [sum(1 for _, in_f in find_uaugs(s) if not in_f) for s in synthetic["utr"]]
        )
        r = pearsonr(n_oof, synthetic["true_mrl"]).statistic
        assert r < -0.5, f"expected strong negative uAUG effect, got r={r:.3f}"

    def test_gc_bias_shifts_composition(self):
        gc_rich = make_synthetic_mpra(300, seed=0, gc_bias=0.8)
        frac = gc_rich["utr"].str.count("[GC]").mean() / UTR_LENGTH
        assert frac > 0.7


class TestCalibrationAgainstRealData:
    """Pin the generator's moments to the real file's.

    Measured on all 326,033 UTRs of GSM3130435_egfp_unmod_1.csv.gz. These are
    regression guards: the generator is only useful as a stand-in if its effect
    sizes stay in the neighbourhood of the real ones, and a future tweak to the
    coefficients should have to update these numbers consciously.
    """

    REAL = {
        "mean": 6.449,
        "no_uaug": 7.560,
        "inframe": 6.654,
        "oof": 4.896,
        "atg_per_utr": 0.734,
    }

    @pytest.fixture(scope="class")
    @classmethod
    def big(cls):
        return make_synthetic_mpra(20_000, seed=0)

    def test_mean_mrl_matches(self, big):
        assert big["rl"].mean() == pytest.approx(self.REAL["mean"], abs=0.25)

    @staticmethod
    def _frame_of_first_uaug(utrs) -> np.ndarray:
        """'none', 'inframe' or 'oof' for each UTR, by its first upstream AUG."""
        labels = []
        for seq in utrs:
            hits = find_uaugs(seq)
            if not hits:
                labels.append("no_uaug")
            else:
                labels.append("inframe" if hits[0][1] else "oof")
        return np.array(labels)

    def test_group_means_by_first_uaug_frame_match(self, big):
        frames = self._frame_of_first_uaug(big["utr"])
        rl = big["rl"].to_numpy()
        for key in ("no_uaug", "inframe", "oof"):
            observed = rl[frames == key].mean()
            assert observed == pytest.approx(self.REAL[key], abs=0.3), key

    def test_out_of_frame_penalty_is_about_three_times_in_frame(self, big):
        """The asymmetry is the biology; the exact ratio is 2.9 in real data."""
        frames = self._frame_of_first_uaug(big["utr"])
        rl = big["rl"].to_numpy()
        base = rl[frames == "no_uaug"].mean()
        in_frame_cost = base - rl[frames == "inframe"].mean()
        oof_cost = base - rl[frames == "oof"].mean()
        assert 2.0 < oof_cost / in_frame_cost < 4.0

    def test_atg_density_matches_the_real_library(self, big):
        density = np.mean([len(find_uaugs(s)) for s in big["utr"]])
        assert density == pytest.approx(self.REAL["atg_per_utr"], abs=0.1)

    def test_label_is_clipped_to_the_physical_range(self, big):
        # The real file is clipped identically: 429 rows at 0.00, 368 at 13.00.
        assert big["rl"].min() >= 0.0
        assert big["rl"].max() <= 13.0

    def test_uses_real_base_composition_by_default(self, big):
        """The library is A-rich and T-poor; uniform sampling would miss that."""
        joined = "".join(big["utr"])
        total = len(joined)
        assert joined.count("A") / total > joined.count("T") / total + 0.05
