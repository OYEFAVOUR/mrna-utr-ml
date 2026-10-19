# mrna-utr-ml

Predict the translation efficiency of a 5'UTR from sequence, then design better ones.

Given a 5'UTR, the model predicts its **mean ribosome load (MRL)** — how many ribosomes an mRNA
carrying that UTR loads, and therefore how much protein it makes per molecule. A constrained
optimiser then proposes new UTRs with higher predicted MRL, which is the inner loop of mRNA
construct design.

> **Status:** in active development. Results are filled in at Stage 8.

<p align="center">
  <img src="docs/figures/overview.svg" alt="Four-stage loop: the mRNA construct, the polysome-profiling measurement that yields mean ribosome load, the sequence-to-MRL model, and the constrained optimiser whose proposed 5'UTRs feed back into the construct. Below, the sequence features the model has to learn: out-of-frame and in-frame upstream AUGs, GC content, and Kozak context." width="100%">
</p>

---

## Results

| Model | Pearson r | r² | Spearman ρ | RMSE |
|---|---|---|---|---|
| Mean predictor (baseline) | — | — | — | — |
| GC-only linear | — | — | — | — |
| k-mer + XGBoost | — | — | — | — |
| 1D CNN | — | — | — | — |

Held out on the same 20,000-UTR test set as Sample et al. 2019, whose Optimus 5-Prime model reaches
r² ≈ 0.93, so the numbers are directly comparable. Metrics carry bootstrap 95% confidence intervals.

## Install

```bash
conda env create -f environment.yml
conda activate utr-mrl
pip install -e .
```

## Use

```bash
utr-mrl download                      # fetch the MPRA data from GEO (~61 MB, checksummed)
utr-mrl train    --config configs/cnn.yaml
utr-mrl evaluate --model models/cnn.pt
utr-mrl predict  --model models/cnn.pt --fasta my_utrs.fa
utr-mrl design   --model models/cnn.pt --start <50-mer> --gc-min 0.4 --gc-max 0.6
```

`utr-mrl --help` lists every command. Training is config-driven: each run is reproducible from a
committed YAML file.

Runs on a laptop CPU with `--subset` for a smaller slice; see [`scripts/gadi/`](scripts/gadi/) for
PBS job scripts and container notes for full training on NCI Gadi.

## Data

Sample et al. 2019, *Nat Biotechnol* — [GEO GSE114002](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114002),
sample `GSM3130435` (eGFP reporter, unmodified chemistry, 326,033 random 50-nt 5'UTRs).

Raw data is not committed; `utr-mrl download` fetches and checksums it. The test suite runs against a
synthetic dataset with known ground truth, so it needs no download and no network.

## Limitations

The model is trained on random 50-mers, so transfer to native human UTRs or other lengths is an
assumption rather than a result. Designed sequences are computational candidates requiring wet-lab
validation. See [MODEL_CARD.md](MODEL_CARD.md).

## Licence

MIT — see [LICENSE](LICENSE).

## Citation

Sample, P.J., Wang, B., Reid, D.W. et al. Human 5′ UTR design and variant effect prediction from a
massively parallel translation assay. *Nat Biotechnol* **37**, 803–809 (2019).
