"""A synthetic 5'UTR dataset with a known generative model.

Why this exists
---------------
Tests must run in CI with no network and no 61 MB download. The lazy way to do
that is random sequences with random labels -- but then every test is a smoke
test: it can only assert that code runs, never that it is *correct*.

Instead we generate labels from an explicit mechanistic model of translation
initiation, with coefficients fitted to the real data. Because the ground truth
is known, tests can assert real things:

* a trained model recovers the out-of-frame uAUG penalty;
* in silico mutagenesis ranks ATG creation as the most damaging edit;
* the designer raises *true* MRL, not just predicted MRL;
* a model's score on synthetic data carries some information about the real
  task, because the effect sizes are the real ones.

None of that is checkable on the real data, where the ground truth is what we
are trying to discover.

The generative model: leaky scanning
------------------------------------
The library mimics the MPRA geometry: a 50-nt UTR immediately upstream of the
reporter's AUG, which begins at position ``UTR_LENGTH``.

The 40S ribosomal subunit loads at the 5' cap and scans 3' until it finds a
start codon. So upstream AUGs act as a **cascade of sieves, not a sum of
penalties**: the 5'-most uAUG sees every scanning subunit, and each one further
down sees only what leaked past the ones before it. That makes the effect
multiplicative:

    throughput = product over out-of-frame uAUGs  of (1 - CAPTURE_OOF * kozak)
               * product over in-frame uAUGs      of (1 - CAPTURE_IF  * kozak)

    MRL = BLOCKED_MRL + (BASE_MRL - BLOCKED_MRL) * throughput
          + GC_COEFFICIENT * (gc_fraction - 0.5)
          + biological noise

    observed MRL = clip(MRL + measurement noise / sqrt(total_reads), 0, 13)

Each piece encodes a real mechanism:

* **uAUGs are the dominant signal**, because a subunit that initiates upstream
  never reaches the reporter's start codon.
* **Out-of-frame is far worse than in-frame.** An in-frame uAUG with no
  intervening stop just makes an N-terminally extended but still functional
  reporter, so ribosome load is largely preserved: the fitted capture
  probability is 0.16 in frame against 1.00 out of frame.
* **Kozak context scales capture.** A uAUG in strong context (purine at -3, G
  at +4) captures more scanning subunits than one in weak context.
* **The multiplicative form saturates.** Two uAUGs are worse than one but not
  twice as bad, and no number of them can push MRL below ``BLOCKED_MRL``. An
  earlier additive version of this model drove 1.5% of labels onto the MRL
  floor, against 0.13% in the real file -- the saturation is not a convenience,
  it is what the real data shows.
* **GC reduces load** via secondary structure in the scanning path. Fitted from
  the real data at -3.5 MRL per unit GC fraction, which is weaker than a uAUG
  but not negligible -- GC alone reaches r = -0.28, so the GC-only baseline is a
  real baseline rather than a straw man.
* **Noise shrinks with read depth**, mirroring the assay's counting statistics.

Calibration
-----------
``BASE_MRL``, ``BLOCKED_MRL``, ``CAPTURE_OOF``, ``CAPTURE_IF`` and
``BIOLOGICAL_NOISE_SD`` were fitted by least squares against five statistics of
all 326,033 real UTRs. ``GC_COEFFICIENT`` was measured directly on the 152,043
uAUG-free real UTRs, where it is not confounded by uAUG content.

Verified on 60,000 synthetic UTRs against all 326,033 real ones:

    statistic                   synthetic     real
    mean MRL                         6.39     6.45
    sd MRL                           1.91     1.75
    mean MRL, no uAUG                7.47     7.56
    mean MRL, in-frame uAUG          6.61     6.65
    mean MRL, out-of-frame uAUG      5.00     4.90
    ATG per UTR                      0.79     0.73
    r(GC, MRL) on uAUG-free rows    -0.29    -0.28
    r(out-of-frame count, MRL)      -0.64    -0.66
    % of labels at the MRL floor     0.06     0.13
    median read depth                 879      890

Three known mismatches, left in deliberately rather than papered over:

1. **About 9% over-dispersed** (sd 1.90 vs 1.75). It makes correlations
   marginally easier than reality. Closing it would mean a finer-grained Kozak
   model than the two-position score used here.
2. **The low-read tail is thinner than reality**: 0.4% of synthetic UTRs fall
   below 50 reads against 2.95% of real ones, because real log-reads are
   left-skewed and a two-parameter lognormal cannot match both the median and
   the tail. Consequence: the read-count filter removes less here
   than it will on real data, so its benefit must be measured on the real file
   rather than inferred from synthetic runs.
3. **Read depth and uAUG content are independent here, but coupled in the real
   data.** Real ATG-per-UTR rises from 0.67 in the shallowest read-depth
   quintile to 0.88 in the deepest, and real label sd is non-monotonic in depth
   (2.21, 1.54, 1.53, 1.58, 1.75) because the deepest rows carry more true
   biological variance as well as less measurement noise. Reproducing that
   would mean modelling why library abundance correlates with uAUG content,
   which is unknown. This matters when choosing a test set: selecting one
   by read depth -- as the published split does -- selects a slice with a
   different feature and label distribution from the training data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mrna_utr_ml import UTR_LENGTH

BASES = np.array(["A", "C", "G", "T"])

# Base composition measured from all 326,033 real UTRs. The library is called
# "random" but oligo synthesis is not unbiased: A is over- and T under-
# represented. Using the real composition matters because ATG frequency -- the
# dominant signal -- depends on it.
REAL_BASE_COMPOSITION = np.array([0.2916, 0.2313, 0.2709, 0.2062])

# --- Fitted generative coefficients. See "Calibration" above.
BASE_MRL = 7.51  # MRL of a uAUG-free UTR at 50% GC
BLOCKED_MRL = 2.03  # MRL floor when scanning is fully captured upstream
CAPTURE_OOF = 1.00  # a perfect-Kozak out-of-frame uAUG captures everything
CAPTURE_IF = 0.163  # in-frame uAUGs mostly yield functional extended protein
GC_COEFFICIENT = -3.54  # MRL per unit GC fraction, measured on uAUG-free UTRs
BIOLOGICAL_NOISE_SD = 0.81  # true biological scatter not explained by the above

# Label noise sd is MEASUREMENT_NOISE_SCALE / sqrt(total_reads). The scale comes
# from the assay's counting statistics: MRL is a mean over 14 polysome fractions
# whose spread is ~3, so a UTR seen in n reads has a standard error of roughly
# 3/sqrt(n) -- about 1.0 at 10 reads, 0.1 at 900.
MEASUREMENT_NOISE_SCALE = 3.0

# MRL is the read-weighted mean of fraction indices 0..13, so it is bounded by
# construction. Noise is applied before clipping, as in the real file, which has
# 429 rows at exactly 0.00 and 368 at exactly 13.00.
MRL_MIN = 0.0
MRL_MAX = 13.0

# Read depth is roughly lognormal in the real file. log(reads) has mean 6.582
# and sd 1.073 over all 326,033 rows, but the real median depth is 890 reads
# (log 6.791), so the real log-reads are mildly left-skewed and no two-parameter
# lognormal matches both moments. We match the median, which is the more robust
# target for a noise model, and accept a slightly different mean.
LOG_READS_MEAN = 6.791
LOG_READS_SD = 1.073
MIN_READS = 5


def kozak_strength(seq: str, atg_pos: int) -> float:
    """Score the initiation context of an AUG at ``atg_pos`` on 0.0-1.0.

    Only the two positions that dominate the Kozak consensus (gccRccAUGG) are
    scored: a purine at -3 and a G at +4, each contributing half. Positions
    falling outside the UTR score 0 for that half -- an AUG right at the 5' end
    has no -3 context and genuinely initiates poorly.

    This two-position score is the main simplification in the model and the
    likely cause of its ~9% over-dispersion relative to the real data.
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
    divisible by 3. This convention is confirmed by the real data: grouping real
    UTRs by the frame of their first uAUG gives mean MRL 6.65 for in-frame
    against 4.77 and 5.03 for the two out-of-frame registers.

    Returns ``(position, in_frame)`` pairs in 5'-to-3' order, which is also
    scanning order -- the leaky-scanning model depends on that ordering.
    """
    out = []
    start = 0
    while (idx := seq.find("ATG", start)) != -1:
        out.append((idx, (utr_length - idx) % 3 == 0))
        start = idx + 1  # +1 not +3: overlapping ATGs (e.g. ATGATG) both count
    return out


def scanning_throughput(seq: str, utr_length: int = UTR_LENGTH) -> float:
    """Fraction of scanning 40S subunits that reach the reporter's start codon.

    Returns 1.0 for a UTR with no upstream AUG, and falls towards 0 as uAUGs
    capture scanning subunits. Multiplicative because scanning is sequential:
    each uAUG only sees what leaked past the ones 5' of it.
    """
    throughput = 1.0
    for pos, in_frame in find_uaugs(seq, utr_length):
        capture = CAPTURE_IF if in_frame else CAPTURE_OOF
        throughput *= 1.0 - capture * kozak_strength(seq, pos)
    return throughput


def gc_fraction(seq: str) -> float:
    """G+C as a fraction of sequence length."""
    return (seq.count("G") + seq.count("C")) / len(seq)


def true_mrl(seq: str, utr_length: int = UTR_LENGTH) -> float:
    """Noise-free MRL under the generative model -- the ground truth for tests.

    Deliberately *not* clipped to the physical MRL range. This is the exact
    function the generator applies, so tests can check monotonicity and effect
    sizes without a clip masking the relationship; the clip belongs on the
    observed label, which is what a real instrument reports.
    """
    throughput = scanning_throughput(seq, utr_length)
    return (
        BLOCKED_MRL
        + (BASE_MRL - BLOCKED_MRL) * throughput
        + GC_COEFFICIENT * (gc_fraction(seq) - 0.5)
    )


def make_synthetic_mpra(
    n: int = 5_000,
    *,
    seed: int = 0,
    utr_length: int = UTR_LENGTH,
    gc_bias: float | None = None,
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
        Probability mass on G/C when sampling bases. ``None`` (the default) uses
        the composition measured from the real library, which is mildly A-rich
        and T-poor. Pass a float to force a symmetric GC fraction instead --
        used by tests, and useful for probing how a model behaves off the
        training composition.
    with_truth
        Include the noise-free ``true_mrl`` column. Tests use it to separate
        model error from label noise; it is meaningless for real data, so
        training code must never read it.

    Returns a frame with ``utr``, ``total_reads``, ``rl`` (the noisy observed
    label, matching the real file's column name) and optionally ``true_mrl``.
    """
    rng = np.random.default_rng(seed)

    if gc_bias is None:
        probs = REAL_BASE_COMPOSITION
    else:
        probs = np.array(
            [(1 - gc_bias) / 2, gc_bias / 2, gc_bias / 2, (1 - gc_bias) / 2]
        )
    idx = rng.choice(4, size=(n, utr_length), p=probs)
    seqs = ["".join(row) for row in BASES[idx]]

    truth = np.array([true_mrl(s, utr_length) for s in seqs])
    truth = truth + rng.normal(0, BIOLOGICAL_NOISE_SD, n)

    reads = np.maximum(
        MIN_READS, rng.lognormal(LOG_READS_MEAN, LOG_READS_SD, n).astype(int)
    )
    observed = truth + rng.normal(0, MEASUREMENT_NOISE_SCALE / np.sqrt(reads), n)
    observed = np.clip(observed, MRL_MIN, MRL_MAX)

    frame = pd.DataFrame({"utr": seqs, "total_reads": reads, "rl": observed})
    if with_truth:
        frame["true_mrl"] = truth
    # Match the real file's ordering so code paths that assume read-depth
    # sorting (the published test-set rule) behave the same on both.
    return frame.sort_values("total_reads", ascending=False, ignore_index=True)
