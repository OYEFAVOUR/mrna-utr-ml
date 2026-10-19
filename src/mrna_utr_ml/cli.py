"""Command line interface for mrna-utr-ml.

Subcommands are added as each stage of the project lands:

    download   fetch the Sample et al. 2019 MPRA data from GEO   (stage 2)
    synth      write a synthetic dataset with known ground truth (stage 2)
    split      build reproducible train/val/test splits           (stage 3)
    train      train a model from a YAML config                  (stage 7)
    evaluate   score a trained model on the held-out test set    (stage 8)
    interpret  in silico mutagenesis to expose learned motifs    (stage 9)
    predict    score new UTR sequences                           (stage 6)
    design     optimise a UTR for predicted MRL under constraints(stage 10)
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mrna_utr_ml import __version__
from mrna_utr_ml.data.download import GEO_URL, fetch_mpra
from mrna_utr_ml.data.load import load_mpra
from mrna_utr_ml.data.split import (
    SplitStrategy,
    assert_no_leakage,
    describe_shift,
    make_splits,
    write_manifest,
)
from mrna_utr_ml.data.synthetic import make_synthetic_mpra

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Predict 5'UTR mean ribosome load from sequence, and design optimised 5'UTRs.",
)


@app.callback()
def main() -> None:
    """Group callback.

    Typer collapses a single-command app into a bare command with no subcommand
    name, so `utr-mrl version` would fail until a second command is registered.
    Declaring a callback forces subcommand mode permanently, which keeps the CLI
    surface stable as commands are added stage by stage.
    """


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo(f"mrna-utr-ml {__version__}")


@app.command()
def download(
    dest: Annotated[
        Path, typer.Option(help="Directory to download into.")
    ] = Path("data/raw"),
    force: Annotated[
        bool, typer.Option(help="Re-download even if the file is already present.")
    ] = False,
) -> None:
    """Download the Sample et al. 2019 5'UTR MPRA data from GEO (~61 MB).

    The file is checksummed and the hash recorded in CHECKSUMS.txt, so repeat
    runs verify your copy rather than assuming it. Interrupted transfers resume.
    """
    typer.echo(f"source: {GEO_URL}")
    result = fetch_mpra(dest)
    typer.echo(
        f"{'reused' if result.reused else 'downloaded'}: {result.path} "
        f"({result.n_bytes / 1e6:.1f} MB)"
    )
    typer.echo(f"sha256: {result.sha256}")


@app.command()
def synth(
    out: Annotated[Path, typer.Option(help="Output csv path.")] = Path(
        "data/processed/synthetic.csv"
    ),
    n: Annotated[int, typer.Option(help="Number of UTRs to generate.")] = 5000,
    seed: Annotated[int, typer.Option(help="RNG seed; fixes the output exactly.")] = 0,
) -> None:
    """Write a synthetic MPRA dataset with a known generative ground truth.

    Used by the test suite so it runs offline, and useful for a fast end-to-end
    sanity check of training before committing to the real data.
    """
    frame = make_synthetic_mpra(n, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    typer.echo(f"wrote {len(frame)} UTRs to {out}")
    typer.echo(
        f"observed MRL range {frame['rl'].min():.2f}-{frame['rl'].max():.2f}; "
        f"read depth {frame['total_reads'].min()}-{frame['total_reads'].max()}"
    )


@app.command()
def split(
    data: Annotated[Path, typer.Option(help="MPRA csv(.gz) to split.")] = Path(
        "data/raw/GSM3130435_egfp_unmod_1.csv.gz"
    ),
    strategy: Annotated[
        SplitStrategy,
        typer.Option(help="`published` matches Sample et al.; `random` avoids its shift."),
    ] = SplitStrategy.random,
    test_size: Annotated[int, typer.Option(help="Target test rows.")] = 20_000,
    val_size: Annotated[int, typer.Option(help="Target validation rows.")] = 20_000,
    seed: Annotated[int, typer.Option(help="Seed; fixes the split exactly.")] = 0,
    manifest: Annotated[
        Path, typer.Option(help="Where to write the split manifest.")
    ] = Path("data/processed/split_manifest.json"),
) -> None:
    """Build train/validation/test splits and report any distribution shift.

    Near-identical sequences are grouped and split as whole groups, so a
    one-nucleotide variant of a training UTR can never appear in the test set.
    """
    frame = load_mpra(data)
    typer.echo(f"loaded {len(frame):,} UTRs from {data}")

    collapsed, result = make_splits(
        frame, strategy=strategy, test_size=test_size, val_size=val_size, seed=seed
    )
    assert_no_leakage(collapsed, result)

    typer.echo(
        f"exact duplicates collapsed: {result.n_exact_duplicates_collapsed}; "
        f"{result.n_grouped_sequences:,} sequences fall in "
        f"{len(collapsed) - result.n_groups:,} near-duplicate merges"
    )
    typer.echo(f"sizes: {result.sizes()}")
    typer.echo("")
    typer.echo(describe_shift(collapsed, result).to_string(index=False))

    written = write_manifest(collapsed, result, manifest)
    typer.echo("")
    typer.echo(f"manifest -> {manifest}  (test digest {written['digests']['test'][:16]}...)")


if __name__ == "__main__":
    app()
