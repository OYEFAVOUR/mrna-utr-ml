"""Load and validate the MPRA table, real or synthetic.

Keeping loading separate from downloading means every later stage -- splitting,
features, training -- takes a validated DataFrame and never needs to know
whether it came from GEO or from the generator. Tests exercise the same code
path as a real run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mrna_utr_ml import UTR_LENGTH
from mrna_utr_ml.data.download import SubsetMode

REQUIRED_COLUMNS = ("utr", "total_reads", "rl")
VALID_BASES = set("ACGT")


class DataValidationError(ValueError):
    """Raised when the loaded table is not shaped the way downstream code assumes."""


def validate_mpra(frame: pd.DataFrame, *, utr_length: int = UTR_LENGTH) -> None:
    """Fail loudly and specifically on a malformed table.

    Every check here corresponds to an assumption some later module makes. A
    wrong-length UTR silently breaks the fixed-width one-hot encoder; a lowercase
    or 'N'-containing sequence silently breaks uAUG counting (``find('ATG')``
    misses ``atg``). Catching these at load time turns a mysteriously poor r²
    into a clear error message.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise DataValidationError(
            f"missing required column(s) {missing}; got {list(frame.columns)}"
        )
    if frame.empty:
        raise DataValidationError("table is empty")

    lengths = frame["utr"].str.len().unique()
    if len(lengths) != 1 or lengths[0] != utr_length:
        raise DataValidationError(
            f"expected every UTR to be {utr_length} nt, found lengths {sorted(lengths)}"
        )

    alphabet = set("".join(frame["utr"].head(1000)))
    if not alphabet <= VALID_BASES:
        raise DataValidationError(
            f"UTRs contain unexpected characters {sorted(alphabet - VALID_BASES)}; "
            "sequences must be uppercase ACGT"
        )
    if frame["rl"].isna().any():
        raise DataValidationError("`rl` contains missing values")
    if (frame["total_reads"] <= 0).any():
        raise DataValidationError("`total_reads` contains non-positive values")


def load_mpra(
    path: str | Path,
    *,
    subset: int | None = None,
    subset_mode: SubsetMode = SubsetMode.head,
    seed: int = 0,
    utr_length: int = UTR_LENGTH,
    validate: bool = True,
) -> pd.DataFrame:
    """Read an MPRA csv(.gz) and return only the columns later stages use.

    The real file carries 32 columns, 28 of which are the per-fraction polysome
    distribution used to derive ``rl``. We keep ``utr``, ``total_reads`` and
    ``rl``: loading the rest would quadruple memory for no modelling gain.

    ``subset`` trims the table for laptop runs. Note that ``SubsetMode.head``
    exploits the file's read-depth sorting and therefore returns an unusually
    clean slice -- see ``SubsetMode`` for why that flatters your metrics.
    """
    usecols = lambda c: c in REQUIRED_COLUMNS or c == "true_mrl"  # noqa: E731
    frame = pd.read_csv(path, usecols=usecols)

    if subset is not None and subset < len(frame):
        if subset_mode is SubsetMode.head:
            frame = frame.head(subset).reset_index(drop=True)
        else:
            frame = frame.sample(subset, random_state=seed).reset_index(drop=True)

    if validate:
        validate_mpra(frame, utr_length=utr_length)
    return frame
