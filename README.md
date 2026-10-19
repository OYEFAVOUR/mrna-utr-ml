# mrna-utr-ml

Predict the translation efficiency of a 5'UTR from sequence, then design better ones.

> **Status:** in active development — the results table is populated as models are trained.

<p align="center">
  <img src="docs/figures/overview.svg" alt="Four-stage loop: the mRNA construct, the polysome-profiling measurement that yields mean ribosome load, the sequence-to-MRL model, and the constrained optimiser whose proposed 5'UTRs feed back into the construct. Below, the sequence features the model has to learn: out-of-frame and in-frame upstream AUGs, GC content, and Kozak context." width="100%">
</p>

---

## Why this matters

In an mRNA therapeutic the coding sequence is fixed by the protein you need to make. The 5'UTR is
not. It largely determines how many ribosomes load onto each transcript, and therefore how much
protein every delivered mRNA molecule produces. More protein per molecule means a lower dose, which
means less lipid nanoparticle, less reactogenicity and lower cost of goods — so the 5'UTR is one of
the cheapest levers available in construct design, and one of the few that can be optimised
computationally.

The quantity that captures this is **mean ribosome load (MRL)**: the average number of ribosomes
riding a transcript, measured by separating mRNAs on a polysome gradient and sequencing each
fraction. A UTR with MRL ≈ 9 is translated hard; one with MRL ≈ 2 is largely ignored. The dominant
sequence determinant is upstream AUGs — a ribosome that starts at one never reaches the real start
codon — modulated by their reading frame, their Kozak context, and secondary structure in the
scanning path.

Measuring MRL for a candidate UTR costs a wet-lab round. Predicting it costs milliseconds, which
turns UTR selection into a search problem. This project trains a model on 326,033 measured UTRs,
checks by in silico mutagenesis that it has learned initiation biology rather than dataset
artefacts, and then uses it to propose new sequences with higher predicted MRL under
manufacturability constraints.

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
