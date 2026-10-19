"""A synthetic 5'UTR dataset with a known generative model.

Why this exists
---------------
Tests must run in CI with no network and no 61 MB download. The lazy way to do
that is random sequences with random labels -- but then every test is a smoke
test: it can only assert that code runs, never that it is *correct*.

Instead we generate labels from an explicit, documented model of translation
initiation. Because the ground truth is known, tests can assert real things:

* a trained model recovers a negative coefficient for out-of-frame uAUGs;
* in silico mutagenesis ranks ATG creation as the most damaging edit;
* filtering on read depth measurably raises achievable correlation;
* the designer raises true MRL, not just predicted MRL.

None of that is checkable on the real data, where the ground truth is what we
are trying to discover.

The generative model
--------------------
The library mimics the MPRA geometry: a 50-nt UTR sitting immediately upstream
of the reporter's AUG, which begins at position ``UTR_LENGTH``.

    MRL = BASE
          - OOF_UAUG_PENALTY * sum(kozak strength of each out-of-frame uAUG)
          - IF_UAUG_PENALTY  * sum(kozak strength of each in-frame uAUG)
          + GC_COEFFICIENT   * (gc_fraction - 0.5)
          + biological noise
    observed MRL = MRL + measurement noise / sqrt(total_reads)

Each term encodes a real mechanism:

* **uAUGs are the dominant signal.** A scanning 40S subunit that initiates at an
  upstream AUG is captured: it translates a short upstream ORF instead of the
  reporter. This is the strongest sequence determinant in the real data.
* **Out-of-frame is worse than in-frame.** An in-frame uAUG with no intervening
  stop simply makes an N-terminally extended but still functional reporter, so
  ribosome load is largely preserved. An out-of-frame uAUG puts the ribosome in
  the wrong register and the reporter is never made.
* **Kozak context scales the penalty.** A uAUG in strong context (purine at -3,
  G at +4) captures more scanning ribosomes than one in weak context, so the
  penalty is weighted by context strength rather than being a flat count.
* **GC has a mild negative effect** via structure in the scanning path --
  secondary structure impedes the 40S subunit. Deliberately weak, so a model
  that only learns GC does poorly: that is what makes the GC-only baseline a
  meaningful thing to beat.
* **Noise shrinks with read depth.** Label noise scales as 1/sqrt(total_reads),
  mirroring the counting statistics of the real assay. This is what makes the
  Stage 4 read-count filter testable rather than taken on faith.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mrna_utr_ml import UTR_LENGTH

BASES = np.array(["A", "C", "G", "T"])

# --- Generative coefficients. Magnitudes are chosen so the resulting MRL range
# --- and the relative importance of the terms resemble the real data (MRL
# --- roughly 2-9, uAUGs dominant), not fitted to it.
BASE_MRL = 6.5
OOF_UAUG_PENALTY = 1.9
IF_UAUG_PENALTY = 0.55
GC_COEFFICIENT = -1.2
BIOLOGICAL_NOISE_SD = 0.25
# Label noise sd is MEASUREMENT_NOISE_SCALE / sqrt(total_reads). The scale is set
# from the real assay's counting statistics: MRL is a mean over 14 polysome
# fractions whose spread is ~3, so a UTR seen in n reads has a standard error of
# roughly 3/sqrt(n) -- about 1.0 at 10 reads, 0.03 at 10,000.
MEASUREMENT_NOISE_SCALE = 3.0

# MRL is the read-weighted mean of fraction indices 0..13, so it is bounded by
# construction -- it cannot be negative or exceed 13. Noise is applied before
# clipping, which is what the real assay does too: a shallow, badly measured UTR
# piles up against the floor rather than going physically impossible.
MRL_MIN = 0.0
MRL_MAX = 13.0

# Read depth is drawn lognormally: most UTRs are shallow, a few are very deep,
# as in the real library (which spans ~10 to ~12,000 reads).
LOG_READS_MEAN = 4.0
LOG_READS_SD = 1.1
MIN_READS = 5


def kozak_strength(seq: str, atg_pos: int) -> float:
    """Score the initiation context of an AUG at ``atg_pos`` on 0.0-1.0.

    Only the two positions that dominate the Kozak consensus
    (gccRccAUGG) are scored: a purine at -3 and a G at +4. Each contributes
    half. Positions falling outside the UTR score 0 for that half -- an AUG
    right at the 5' end has no -3 context and genuinely initiates poorly.
    """
    score = 0.0
    minus3 = atg_pos - 3
    if minus3 >= 0 and seq[minus3] in "AG":
        score += 0.5
    plus4 = atg_pos + 3
    if plus4 < len(seq) and seq[plus4] == "G":
        score += 0.5
    return score


def find_uaugs(seq: str, utr_length: int = UTR_LENGTH) -> list[tuple[int, bool]]:
    """Locate every upstream ATG and say whether it is in frame with the reporter.

    The reporter's own AUG starts at index ``utr_length`` (just past the UTR), so
    an ATG at index ``i`` shares its reading frame when ``(utr_length - i)`` is
    divisible by 3.

    Returns a list of ``(position, in_frame)``.
    """
    out = []
    start = 0
    while (idx := seq.find("ATG", start)) != -1:
        out.append((idx, (utr_length - idx) % 3 == 0))
        start = idx + 1  # +1 not +3: overlapping ATGs (e.g. ATGATG) both count
    return out


def true_mrl(seq: str, utr_length: int = UTR_LENGTH) -> float:
    """Noise-free MRL under the generative model -- the ground truth for tests.

    Deliberately *not* clipped to the physical MRL range. This is the exact
    linear function the generator applies, so tests can check coefficients and
    monotonicity without a clip masking the relationship; the clip belongs on
    the observed label, which is what a real instrument reports.
    """
    oof = sum(
        kozak_strength(seq, pos) for pos, in_frame in find_uaugs(seq, utr_length) if not in_frame
    )
    in_f = sum(
        kozak_strength(seq, pos) for pos, in_frame in find_uaugs(seq, utr_length) if in_frame
    )
    gc = (seq.count("G") + seq.count("C")) / len(seq)
    return (
        BASE_MRL
        - OOF_UAUG_PENALTY * oof
        - IF_UAUG_PENALTY * in_f
        + GC_COEFFICIENT * (gc - 0.5)
    )


def make_synthetic_mpra(
    n: int = 5_000,
    *,
    seed: int = 0,
    utr_length: int = UTR_LENGTH,
    gc_bias: float = 0.5,
    with_truth: bool = True,
) -> pd.DataFrame:
    """Generate a synthetic MPRA table with the same columns we use downstream.

    Parameters
    ----------
    n
        Number of UTRs.
    seed
        Seeds a ``default_rng``; identical ``(n, seed)`` always gives an
        identical table, so a failing test is reproducible.
    gc_bias
        Probability mass on G/C when sampling bases. 0.5 gives a uniform
        library like the real random 50-mers.
    with_truth
        Include the noise-free ``true_mrl`` column. Tests use it to separate
        model error from label noise; it is meaningless for real data, so
        training code must never read it.

    Returns a frame with ``utr``, ``total_reads``, ``rl`` (the noisy observed
    label, matching the real file's column name) and optionally ``true_mrl``.
    """
    rng = np.random.default_rng(seed)

    probs = np.array([(1 - gc_bias) / 2, gc_bias / 2, gc_bias / 2, (1 - gc_bias) / 2])
    idx = rng.choice(4, size=(n, utr_length), p=probs)
    seqs = ["".join(row) for row in BASES[idx]]

    truth = np.array([true_mrl(s, utr_length) for s in seqs])
    truth = truth + rng.normal(0, BIOLOGICAL_NOISE_SD, n)

    reads = np.maximum(
        MIN_READS,
        rng.lognormal(LOG_READS_MEAN, LOG_READS_SD, n).astype(int),
    )
    observed = truth + rng.normal(0, MEASUREMENT_NOISE_SCALE / np.sqrt(reads), n)
    observed = np.clip(observed, MRL_MIN, MRL_MAX)

    frame = pd.DataFrame({"utr": seqs, "total_reads": reads, "rl": observed})
    if with_truth:
        frame["true_mrl"] = truth
    # Match the real file's ordering so code paths that assume read-depth
    # sorting (the published test-set rule) behave the same on both.
    return frame.sort_values("total_reads", ascending=False, ignore_index=True)
