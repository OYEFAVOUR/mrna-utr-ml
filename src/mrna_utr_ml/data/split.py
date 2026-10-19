"""Train/validation/test splitting for the MPRA data.

Two separate problems are solved here, and they are worth keeping apart.

1. Leakage between near-identical sequences
-------------------------------------------
If two near-identical UTRs land either side of a split, the test score is partly
a memory test. The usual defence is to group similar sequences and split by
group, never by row.

I expected this to be a formality on a random 50-mer library. It is not. There
are zero exact duplicates in 326,033 rows, but MinHash grouping at Jaccard 0.6
finds **6,324 near-duplicate groups covering 12,689 sequences (3.9% of the
library)**, and they are not chance:

* 5,974 of 6,284 pairs differ by exactly **one** nucleotide;
* that base sits at position 0 in 56% of cases and position 1 in a further 25%
  -- a **28x enrichment** at position 0 over the uniform expectation;
* the shallower member of a pair has a median of 92 reads against 890 for the
  library, and 34% of them fall below 50 reads against 2.95% library-wide.

That is the signature of a 5'-end synthesis or sequencing error producing a
spurious second library member from one real oligo, not of two independently
synthesised UTRs. Their apparent disagreement in MRL is mostly the shallow
member being badly measured: mean |MRL difference| within a pair falls from
5.20 when the shallower member has under 20 reads to 2.45 at 200-1,000 reads.
(It does not fall all the way to the 1.94 of random pairs, which is not fully
explained -- a change at position 0 or 1 can also alter the cap-proximal
context, so some of the residual may be real.)

**Consequence:** on a row-wise split roughly 724 of these pairs would straddle
train and test, putting a one-nucleotide variant of a training sequence into the
test set. Grouping is therefore doing real work here, not ceremony.

Sequences are grouped rather than merged. Merging would assert they are the same
molecule, which the read-depth evidence supports but does not prove; grouping
only asserts they must not be scored against each other, which is enough.

2. Which test set to report
---------------------------
Sample et al. hold out the 20,000 UTRs with the highest read depth. Using the
same rule makes our numbers directly comparable to their published r2, so we
support it. But it is not a neutral sample of the library:

    read-depth quintile   shallowest  ->  deepest
    ATG per UTR                 0.67  ...  0.88

The deepest rows carry ~30% more upstream AUGs than the shallowest, and uAUGs
are the dominant driver of MRL. So the published test set is **shifted** in both
feature and label distribution relative to what the model trains on. A score on
it answers "how well do we do on unusually deep, unusually uAUG-rich UTRs",
which is not the same question as "how well will this generalise".

Both splits are therefore first-class, and the gap between them is reported
rather than hidden. ``SplitStrategy.published`` for comparability,
``SplitStrategy.random`` for an unbiased estimate of generalisation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd

BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}
# Mersenne prime 2^31 - 1. Small enough that a * kmer_id stays well inside
# int64 for vectorised hashing, large enough that collisions are rare -- and
# every candidate pair is verified exactly afterwards, so a collision costs
# time, never correctness.
_HASH_PRIME = 2_147_483_647


class SplitStrategy(StrEnum):
    """How the held-out test set is chosen.

    ``published`` the 20,000 highest-read-depth UTRs, matching Sample et al.
                  2019 so metrics are comparable with the literature. Carries
                  the covariate shift described in the module docstring.
    ``random``    a uniform random sample of groups. No shift, so this is the
                  honest estimate of generalisation to the library as a whole.
    """

    published = "published"
    random = "random"


@dataclass(frozen=True)
class SplitResult:
    """Row indices for each split, plus everything needed to reproduce it."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    strategy: SplitStrategy
    seed: int
    n_groups: int
    n_grouped_sequences: int
    n_exact_duplicates_collapsed: int
    jaccard_threshold: float

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


# --------------------------------------------------------------------------
# exact duplicates
# --------------------------------------------------------------------------


def collapse_exact_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Merge rows measuring the same UTR into one.

    The same sequence appearing twice with two noisy MRL estimates is one
    measurement repeated, not two data points: leaving both in would let the
    identical sequence sit on both sides of a split, and would weight it double
    during training.

    MRL is combined as a **read-weighted** mean rather than a plain mean,
    because a 5,000-read estimate deserves more weight than a 10-read one.
    Reads are summed, which is what the combined evidence amounts to.

    Returns the collapsed frame and the number of rows removed.
    """
    n_before = len(frame)
    if not frame["utr"].duplicated().any():
        return frame.reset_index(drop=True), 0

    weighted = frame["rl"] * frame["total_reads"]
    work = frame.assign(_weighted=weighted)
    agg = {"_weighted": "sum", "total_reads": "sum"}
    if "true_mrl" in frame.columns:
        agg["true_mrl"] = "first"  # noise-free, identical across duplicates

    merged = work.groupby("utr", as_index=False, sort=False).agg(agg)
    merged["rl"] = merged["_weighted"] / merged["total_reads"]
    merged = merged.drop(columns="_weighted")
    merged = merged[[c for c in frame.columns if c in merged.columns]]
    merged = merged.sort_values("total_reads", ascending=False, ignore_index=True)
    return merged, n_before - len(merged)


# --------------------------------------------------------------------------
# near-duplicate grouping: MinHash + LSH + union-find
# --------------------------------------------------------------------------


def kmer_id_matrix(seqs: Sequence[str], k: int = 6) -> np.ndarray:
    """Encode every k-mer of every sequence as an integer in [0, 4**k).

    Returns an ``(n_sequences, n_kmers)`` array. Vectorised because the real
    library is 326,033 x 45 k-mers and a Python loop over that is minutes.
    """
    if not seqs:
        return np.zeros((0, 0), dtype=np.int64)
    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise ValueError(f"expected equal-length sequences, got lengths {sorted(lengths)}")
    length = lengths.pop()
    if k > length:
        raise ValueError(f"k={k} exceeds sequence length {length}")

    codes = np.frombuffer("".join(seqs).encode(), dtype=np.uint8).reshape(len(seqs), length)
    lookup = np.full(256, -1, dtype=np.int64)
    for base, value in BASE_TO_INT.items():
        lookup[ord(base)] = value
    digits = lookup[codes]
    if (digits < 0).any():
        raise ValueError("sequences must contain only A, C, G, T")

    n_kmers = length - k + 1
    ids = np.zeros((len(seqs), n_kmers), dtype=np.int64)
    for offset in range(k):
        ids = ids * 4 + digits[:, offset : offset + n_kmers]
    return ids


def minhash_signatures(
    kmer_ids: np.ndarray, n_perm: int = 128, seed: int = 0
) -> np.ndarray:
    """MinHash signature per sequence.

    The probability that two signatures agree in any one position equals the
    Jaccard similarity of their k-mer sets, which is what makes the banding
    trick below a similarity search rather than a guess.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _HASH_PRIME, n_perm, dtype=np.int64)
    b = rng.integers(0, _HASH_PRIME, n_perm, dtype=np.int64)
    signatures = np.empty((kmer_ids.shape[0], n_perm), dtype=np.int64)
    for j in range(n_perm):
        signatures[:, j] = ((a[j] * kmer_ids + b[j]) % _HASH_PRIME).min(axis=1)
    return signatures


def _band_codes(signatures: np.ndarray, bands: int) -> np.ndarray:
    """Hash each band of the signature matrix to one integer per sequence.

    Sequences sharing a band code are *candidates*; a shared band is evidence of
    similarity, not proof, so every candidate is verified exactly afterwards.
    """
    n_perm = signatures.shape[1]
    if n_perm % bands:
        raise ValueError(f"bands={bands} must divide n_perm={n_perm}")
    rows = n_perm // bands
    codes = np.empty((signatures.shape[0], bands), dtype=np.int64)
    for band in range(bands):
        chunk = signatures[:, band * rows : (band + 1) * rows]
        acc = np.zeros(signatures.shape[0], dtype=np.int64)
        for t in range(rows):
            acc = (acc * 1_000_003 + chunk[:, t]) % _HASH_PRIME
        codes[:, band] = acc
    return codes


class _UnionFind:
    def __init__(self, n: int) -> None:
        self._parent = np.arange(n)

    def find(self, x: int) -> int:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:  # path compression
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)


def group_near_duplicates(
    seqs: Sequence[str],
    *,
    k: int = 6,
    threshold: float = 0.6,
    n_perm: int = 128,
    bands: int = 32,
    seed: int = 0,
    max_bucket: int = 2_000,
) -> np.ndarray:
    """Assign each sequence a group id; similar sequences share a group.

    Similarity is Jaccard overlap of k-mer sets. Candidate pairs come from
    MinHash banding, then each candidate is confirmed by computing the Jaccard
    index exactly, so the output never depends on a hash collision. Confirmed
    pairs are merged transitively with union-find: if A is similar to B and B to
    C, all three must stay on the same side of a split even when A and C are not
    themselves similar.

    ``max_bucket`` caps the size of any one LSH bucket. A bucket of m sequences
    costs m^2 comparisons, so an unexpectedly huge bucket would stall the run;
    on random sequence this never triggers, and if it ever does the right answer
    is to look at the data rather than wait.
    """
    n = len(seqs)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if n == 1:
        return np.zeros(1, dtype=np.int64)

    kmer_ids = kmer_id_matrix(seqs, k)
    signatures = minhash_signatures(kmer_ids, n_perm=n_perm, seed=seed)
    codes = _band_codes(signatures, bands)

    kmer_sets = [set(row.tolist()) for row in kmer_ids]
    union = _UnionFind(n)

    for band in range(codes.shape[1]):
        order = np.argsort(codes[:, band], kind="stable")
        sorted_codes = codes[order, band]
        starts = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1]])
        for start, stop in zip(starts, np.r_[starts[1:], n], strict=True):
            bucket = order[start:stop]
            if len(bucket) < 2 or len(bucket) > max_bucket:
                continue
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    a, b = int(bucket[i]), int(bucket[j])
                    if union.find(a) == union.find(b):
                        continue
                    sa, sb = kmer_sets[a], kmer_sets[b]
                    overlap = len(sa & sb) / len(sa | sb)
                    if overlap >= threshold:
                        union.union(a, b)

    roots = np.array([union.find(i) for i in range(n)])
    _, groups = np.unique(roots, return_inverse=True)
    return groups.astype(np.int64)


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------


def make_splits(
    frame: pd.DataFrame,
    *,
    strategy: SplitStrategy = SplitStrategy.random,
    test_size: int = 20_000,
    val_size: int = 20_000,
    seed: int = 0,
    group: bool = True,
    jaccard_threshold: float = 0.6,
    k: int = 6,
) -> tuple[pd.DataFrame, SplitResult]:
    """Split a loaded MPRA frame into train / validation / test.

    Exact duplicates are collapsed first, then near-duplicates are grouped, then
    whole groups are assigned to splits. Validation is always drawn at random
    from what remains, whichever test strategy is used: tuning against a shifted
    validation set would bake the shift into the model's hyperparameters too.

    Returns the collapsed frame (row indices in ``SplitResult`` refer to it) and
    the split.
    """
    collapsed, n_dupes = collapse_exact_duplicates(frame)
    n = len(collapsed)
    if test_size + val_size >= n:
        raise ValueError(
            f"test_size + val_size ({test_size + val_size}) must be smaller "
            f"than the dataset ({n} rows)"
        )

    if group:
        groups = group_near_duplicates(
            collapsed["utr"].tolist(), k=k, threshold=jaccard_threshold, seed=seed
        )
    else:
        groups = np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(seed)
    # `published` walks the frame in order, which is read-depth descending, so
    # the deepest UTRs land in test. `random` walks a shuffle.
    order = np.arange(n) if strategy is SplitStrategy.published else rng.permutation(n)

    group_sizes = np.bincount(groups)
    test_groups: set[int] = set()
    held = 0
    for idx in order:
        g = int(groups[idx])
        if g in test_groups:
            continue
        test_groups.add(g)
        held += int(group_sizes[g])
        if held >= test_size:
            break

    remaining = np.array([i for i in range(n) if int(groups[i]) not in test_groups])
    val_order = rng.permutation(len(remaining))
    val_groups: set[int] = set()
    held = 0
    for pos in val_order:
        g = int(groups[remaining[pos]])
        if g in val_groups:
            continue
        val_groups.add(g)
        held += int(group_sizes[g])
        if held >= val_size:
            break

    in_test = np.array([int(g) in test_groups for g in groups])
    in_val = np.array([int(g) in val_groups for g in groups]) & ~in_test
    test_idx = np.flatnonzero(in_test)
    val_idx = np.flatnonzero(in_val)
    train_idx = np.flatnonzero(~in_test & ~in_val)

    n_grouped = int((group_sizes[group_sizes > 1]).sum())
    return collapsed, SplitResult(
        train=train_idx,
        val=val_idx,
        test=test_idx,
        strategy=strategy,
        seed=seed,
        n_groups=int(group_sizes.size),
        n_grouped_sequences=n_grouped,
        n_exact_duplicates_collapsed=n_dupes,
        jaccard_threshold=jaccard_threshold,
    )


# --------------------------------------------------------------------------
# reporting and reproducibility
# --------------------------------------------------------------------------


def assert_no_leakage(frame: pd.DataFrame, split: SplitResult) -> None:
    """Fail if any sequence appears in more than one split.

    Cheap, and the failure it catches is one that otherwise shows up only as a
    test score that is too good.
    """
    sets = {
        name: set(frame["utr"].iloc[idx])
        for name, idx in (("train", split.train), ("val", split.val), ("test", split.test))
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = sets[a] & sets[b]
        if shared:
            raise AssertionError(
                f"{len(shared)} sequence(s) appear in both {a} and {b}; "
                f"example: {next(iter(shared))}"
            )


def describe_shift(frame: pd.DataFrame, split: SplitResult) -> pd.DataFrame:
    """Compare the splits on the quantities a shift would move.

    Reported rather than assumed: if the test set differs from train in mean
    MRL, read depth or uAUG content, the headline metric is answering a slightly
    different question and the reader deserves to see that.
    """
    from mrna_utr_ml.data.synthetic import find_uaugs, gc_fraction

    rows = []
    for name, idx in (("train", split.train), ("val", split.val), ("test", split.test)):
        part = frame.iloc[idx]
        uaugs = [find_uaugs(s) for s in part["utr"]]
        rows.append(
            {
                "split": name,
                "n": len(part),
                "mean_mrl": part["rl"].mean(),
                "sd_mrl": part["rl"].std(),
                "median_reads": float(part["total_reads"].median()),
                "uaug_per_utr": float(np.mean([len(u) for u in uaugs])),
                "oof_uaug_per_utr": float(
                    np.mean([sum(1 for _, f in u if not f) for u in uaugs])
                ),
                "mean_gc": float(np.mean([gc_fraction(s) for s in part["utr"]])),
            }
        )
    return pd.DataFrame(rows)


def _sequence_digest(seqs: Sequence[str]) -> str:
    """One hash over a split's sequences, order-independent.

    Sorted before hashing so the digest identifies the *set* of sequences, not
    the order they happened to come out in.
    """
    digest = hashlib.sha256()
    for seq in sorted(seqs):
        digest.update(seq.encode())
    return digest.hexdigest()


def write_manifest(frame: pd.DataFrame, split: SplitResult, path: str | Path) -> dict:
    """Record the split as hashes, not sequences, so it can be committed.

    Committing 326,033 sequences would be a 16 MB blob in git. A digest per
    split is 64 characters and answers the question that actually matters: did
    this run use the same split as that run?
    """
    manifest = {
        **{k: (str(v) if isinstance(v, StrEnum) else v) for k, v in asdict(split).items()
           if not isinstance(v, np.ndarray)},
        "sizes": split.sizes(),
        "digests": {
            name: _sequence_digest(frame["utr"].iloc[idx])
            for name, idx in (
                ("train", split.train),
                ("val", split.val),
                ("test", split.test),
            )
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
