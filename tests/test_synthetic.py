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
    find_uaugs,
    kozak_strength,
    make_synthetic_mpra,
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
        # Real egfp_unmod MRL spans roughly 2-9.
        assert 0 < synthetic["rl"].min() < 4
        assert 5 < synthetic["rl"].max() < 14

    def test_label_noise_shrinks_with_read_depth(self, synthetic):
        """The property that makes the stage-4 read-count filter worth having.

        Deep rows should track the ground truth much better than shallow ones.
        Asserted here so that if the generator is ever changed, the
        justification for filtering does not silently evaporate.
        """
        deep = synthetic.head(400)
        shallow = synthetic.tail(400)
        r_deep = pearsonr(deep["rl"], deep["true_mrl"]).statistic
        r_shallow = pearsonr(shallow["rl"], shallow["true_mrl"]).statistic
        assert r_deep > r_shallow + 0.1

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
