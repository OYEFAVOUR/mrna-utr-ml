# mrna-utr-ml — 5'UTR translation efficiency: predict it, then design it

Predict **mean ribosome load (MRL)** of a human 5'UTR from sequence alone, then use the trained
model to propose optimised 5'UTRs for mRNA constructs.

> **Status:** in active development, built stage by stage. Results table below is filled in at Stage 8.

---

## Why this problem

In an mRNA therapeutic the coding sequence is fixed by the protein you want to make. The 5'UTR is
not — it is a tunable knob, and it largely sets **how many ribosomes load onto each transcript**, and
therefore how much protein you get per mRNA molecule delivered. Getting more protein per molecule
means a lower dose, which means less lipid nanoparticle, less reactogenicity and lower cost of goods.

So the design question is: *given a candidate 5'UTR, how well will it be translated — and can we do
better?*

### What MRL actually is

Sample et al. (2019, *Nature Biotechnology*) answered this at scale with a massively parallel
reporter assay (MPRA). They synthesised ~280,000 **random 50-nucleotide 5'UTRs** upstream of an eGFP
reporter, transfected them as a pool, and separated the transcripts on a **polysome gradient** — a
sucrose gradient that sorts mRNAs by how many ribosomes are sitting on them. Sequencing each
gradient fraction tells you, for every UTR in the library, the distribution of its transcripts across
polysome fractions. The read-weighted average of that distribution is its **mean ribosome load**:

> MRL = Σ (fraction index × reads in that fraction) / Σ (reads)

A UTR with MRL ≈ 8 has transcripts carrying ~8 ribosomes on average and is translated hard; a UTR
with MRL ≈ 2 is poorly translated. It is a single continuous number per sequence.

### Why this is a regression task

The label is a continuous scalar, so this is **regression**, not classification. Two consequences
shape the whole project:

1. **Rank correlation is the headline metric, not RMSE.** The downstream use is *ranking candidate
   UTRs* to pick which few to synthesise and test. What matters is whether the model orders
   sequences correctly, so we report **Pearson r / r²** (comparable to the published literature) and
   **Spearman ρ** (rank-only, robust to any monotone distortion of the MRL scale). RMSE is reported
   too, but a model that is off by a constant offset yet ranks perfectly is still useful; a model
   with great RMSE that ranks badly is not.
2. **The MRL label has measurement noise, and that noise caps achievable r².** A UTR seen in only a
   handful of reads has a badly estimated MRL. We therefore filter on read depth and show the
   variance-vs-depth relationship rather than pretending the ceiling is r² = 1.

### How the output informs mRNA design

The trained model is a cheap *in silico* assay: millions of candidate UTRs per minute, versus weeks
per wet-lab round. That enables two things this repo does:

- **Interpretation** (`utr-mrl interpret`) — in silico saturation mutagenesis recovers the biology
  the model learned. The dominant signal should be that **upstream AUGs (uAUGs) are destructive**,
  especially out-of-frame ones, because a scanning ribosome that initiates at a uAUG never reaches
  the real start codon. Strong Kozak context around a uAUG should make it worse. If the model learned
  that unprompted, it has learned translation initiation, not dataset artefacts.
- **Design** (`utr-mrl design`) — a constrained optimiser that mutates a starting UTR uphill in
  predicted MRL while respecting manufacturability constraints (GC window, no new uAUGs, bounded
  edit distance from the parent sequence).

**Honest caveat, stated up front:** these are computational candidates, not validated sequences. The
model is trained on random 50-mers, so transfer to native human UTRs or to other lengths is an
assumption, not a result. Optimising hard against your own model also walks you off the edge of its
training distribution. See [MODEL_CARD.md](MODEL_CARD.md).

---

## Results

| Model | Test Pearson r | Test r² | Spearman ρ | RMSE |
|---|---|---|---|---|
| Mean predictor (baseline) | — | — | — | — |
| GC-only linear | — | — | — | — |
| k-mer + XGBoost | — | — | — | — |
| 1D CNN | — | — | — | — |

Published reference: Optimus 5-Prime (Sample et al. 2019) reaches r² ≈ 0.93 on the held-out
random-library test set. We hold out the same test set so the numbers are directly comparable.

---

## Quickstart

```bash
conda env create -f environment.yml
conda activate utr-mrl
pip install -e ".[dev]"

pytest                      # runs on synthetic data; no download needed
utr-mrl --help
```

Full documentation of each command and the Gadi HPC workflow follows as the project is built out.

---

## Data

Sample et al. 2019, GEO accession [GSE114002](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114002).
Primary file: `GSM3130435_egfp_unmod_1.csv.gz` (eGFP reporter, unmodified nucleotide chemistry,
random 50-mer library, 61 MB) — the same library Optimus 5-Prime was trained on.

Raw data is **not** committed to this repository; `utr-mrl download` fetches and checksums it. A
synthetic dataset with known ground truth ships with the tests so the suite runs offline.

---

## Licence

MIT — see [LICENSE](LICENSE).

## Citation

Sample, P.J., Wang, B., Reid, D.W. et al. Human 5′ UTR design and variant effect prediction from a
massively parallel translation assay. *Nat Biotechnol* **37**, 803–809 (2019).
