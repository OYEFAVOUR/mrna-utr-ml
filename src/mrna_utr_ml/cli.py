"""Command line interface for mrna-utr-ml.

Subcommands are added as each stage of the project lands:

    download   fetch the Sample et al. 2019 MPRA data from GEO   (stage 2)
    train      train a model from a YAML config                  (stage 7)
    evaluate   score a trained model on the held-out test set    (stage 8)
    interpret  in silico mutagenesis to expose learned motifs    (stage 9)
    predict    score new UTR sequences                           (stage 6)
    design     optimise a UTR for predicted MRL under constraints(stage 10)
"""

import typer

from mrna_utr_ml import __version__

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Predict 5'UTR mean ribosome load from sequence, and design optimised 5'UTRs.",
)


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo(f"mrna-utr-ml {__version__}")


if __name__ == "__main__":
    app()
