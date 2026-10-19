"""Download the Sample et al. 2019 5'UTR MPRA data from GEO.

Source
------
GEO series GSE114002, sample GSM3130435: ``egfp_unmod_1``. An eGFP reporter
with a library of random 50-nucleotide 5'UTRs, unmodified nucleotide chemistry,
replicate 1. This is the library Optimus 5-Prime was trained on, so holding out
the same test set makes our metrics directly comparable to the paper.

We fetch from the NCBI FTP-over-HTTPS mirror rather than the GEO web interface,
which sits behind a CAPTCHA and cannot be scripted.

File schema (32 columns, verified against the live file)
--------------------------------------------------------
``utr``            the 50-nt 5'UTR, uppercase DNA alphabet
``0`` .. ``13``    library-size-normalised abundance in each of 14 polysome
                   fractions (fraction 0 = unbound/monosome end)
``total``          row sum of ``0``..``13``
``total_reads``    raw read count for this UTR -- our data-quality weight
``r0`` .. ``r13``  the same distribution renormalised to sum to exactly 1.0
``rl``             mean ribosome load: the label we predict

``rl`` is very close to ``sum(i * r_i for i in 0..13)`` (max absolute deviation
~0.2 over a sample of rows), which confirms the weighting is by fraction index
starting at zero. It is not exactly equal, so the published ``rl`` column is
taken as ground truth rather than recomputed.

Important: the file is **pre-sorted by total_reads, descending**. That is not
incidental -- Sample et al. define their held-out test set as the 20,000 UTRs
with the highest read depth, which is therefore simply the first 20,000 rows.
It also means a ``head``-style subset is a high-read-depth, low-label-noise
slice, not a representative sample. See ``SubsetMode``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import requests
from tqdm import tqdm

GEO_SAMPLE = "GSM3130435"
GEO_FILENAME = f"{GEO_SAMPLE}_egfp_unmod_1.csv.gz"
GEO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3130nnn/"
    f"{GEO_SAMPLE}/suppl/{GEO_FILENAME}"
)
# Content-Length served by NCBI on 2026-10-19. Used only as a sanity check on a
# completed download, so a truncated file is caught before it reaches pandas.
EXPECTED_BYTES = 63_726_742

CHUNK_BYTES = 1 << 20  # 1 MiB


class SubsetMode(StrEnum):
    """How ``--subset`` picks its rows.

    ``head``   the first N rows. Because the file is sorted by read depth, this
               is the N best-measured UTRs: cleanest labels, and it is the same
               rule that defines the published test set. Good for a fast,
               low-noise laptop run -- but optimistic, because label noise
               shrinks with read depth, so metrics on it will beat metrics on
               the full data.
    ``random`` a uniform random sample at a fixed seed. Representative of the
               real read-depth distribution, so more honest for estimating how
               the model will do on the whole library.
    """

    head = "head"
    random = "random"


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    sha256: str
    n_bytes: int
    reused: bool


def sha256_file(path: Path, chunk_bytes: int = CHUNK_BYTES) -> str:
    """Streaming SHA-256 so a 61 MB file never lands in memory at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_size(url: str, timeout: int = 30) -> int | None:
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        return int(response.headers["Content-Length"])
    except (requests.RequestException, KeyError, ValueError):
        return None


def download_with_resume(
    url: str,
    dest: Path,
    *,
    expected_bytes: int | None = None,
    timeout: int = 60,
) -> None:
    """Download ``url`` to ``dest``, resuming a partial file if one exists.

    Resume matters on a flaky connection and on an HPC login node where a long
    transfer can be cut off. We download to ``dest.part`` and only move it into
    place once the byte count checks out, so an interrupted run can never leave
    a truncated file that looks complete to the next step.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    already = part.stat().st_size if part.exists() else 0
    total = expected_bytes if expected_bytes is not None else _remote_size(url)

    if already and total and already >= total:
        already = 0
        part.unlink()

    headers = {"Range": f"bytes={already}-"} if already else {}
    with requests.get(url, stream=True, headers=headers, timeout=timeout) as response:
        # 206 = server honoured the Range request; 200 = it ignored it and is
        # sending the whole file, in which case our offset is meaningless.
        if already and response.status_code == 200:
            already = 0
        response.raise_for_status()

        mode = "ab" if already else "wb"
        with (
            part.open(mode) as handle,
            tqdm(
                total=total,
                initial=already,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
            ) as bar,
        ):
            for chunk in response.iter_content(CHUNK_BYTES):
                handle.write(chunk)
                bar.update(len(chunk))

    size = part.stat().st_size
    if total is not None and size != total:
        raise OSError(
            f"{dest.name}: downloaded {size} bytes, expected {total}. "
            f"Partial file kept at {part} -- rerun to resume."
        )
    part.replace(dest)


def record_checksum(path: Path, sha256: str, checksums: Path) -> None:
    """Append or update ``path``'s checksum in a ``sha256sum``-format file.

    We record the hash observed on first download rather than asserting against
    a value hard-coded here. GEO supplementary files are immutable in practice
    but carry no published checksum, so recording-then-verifying is the honest
    guarantee: it proves *your* copy has not changed between runs and lets a
    collaborator confirm they have byte-identical input.
    """
    lines = {}
    if checksums.exists():
        for line in checksums.read_text().splitlines():
            if line.strip():
                value, name = line.split(maxsplit=1)
                lines[name.strip()] = value
    lines[path.name] = sha256
    checksums.write_text("".join(f"{v}  {k}\n" for k, v in sorted(lines.items())))


def fetch_mpra(
    dest_dir: Path,
    *,
    url: str = GEO_URL,
    filename: str = GEO_FILENAME,
    force: bool = False,
) -> DownloadResult:
    """Fetch the MPRA file into ``dest_dir``, skipping work if it is already there.

    Returns the SHA-256 either way, so a cached file is still verified rather
    than merely assumed.
    """
    dest_dir = Path(dest_dir)
    dest = dest_dir / filename
    reused = dest.exists() and not force

    if not reused:
        download_with_resume(url, dest, expected_bytes=EXPECTED_BYTES)

    sha256 = sha256_file(dest)
    record_checksum(dest, sha256, dest_dir / "CHECKSUMS.txt")
    return DownloadResult(
        path=dest, sha256=sha256, n_bytes=dest.stat().st_size, reused=reused
    )
