# Model card — mrna-utr-ml

> **Status:** placeholder. Trained models do not exist yet; metrics, intended-use
> and fairness sections are completed at Stage 8 and Stage 11. What follows is
> true of the data and the splits as of Stage 3.

## Model details

Predicts mean ribosome load (MRL) of a 50-nucleotide 5'UTR from sequence alone.
Planned: a mean-predictor baseline, a GC-only linear baseline, a k-mer +
XGBoost model, and a 1D CNN. Not yet trained.

## Intended use

Ranking candidate 5'UTR sequences during mRNA construct design, and generating
candidate UTRs for wet-lab validation.

**Out of scope.** Not for clinical decision-making. Not validated on native
human 5'UTRs, on lengths other than 50 nt, on modified nucleotide chemistries,
or in cell types other than the reporter system used in the source assay.

## Training data

Sample et al. 2019, *Nat Biotechnol*, GEO
[GSE114002](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114002),
sample `GSM3130435` — eGFP reporter, unmodified chemistry, 326,033 random 50-nt
5'UTRs measured by polysome profiling. SHA-256
`23bbd468ff6c6905f11e7dfdd7509601730e0f99c8ad2a78f37f3dfe99c31719`.

The library is a **random** 50-mer library, not a sample of native human UTRs.
Any claim about human UTRs is therefore extrapolation.

## Evaluation

Two held-out test sets are reported, because they answer different questions.

- **Published split** — the 20,000 highest-read-depth UTRs, matching Sample et
  al. so metrics compare directly with Optimus 5-Prime (r² ≈ 0.93). This test
  set is *not* representative: it has 3.6× the median read depth of the training
  data, 44% more upstream AUGs per UTR, and mean MRL 0.78 lower.
- **Random split** — groups sampled uniformly. Matches the training distribution
  on read depth, uAUG content, GC and MRL, so it is the honest estimate of
  generalisation to the library.

Metrics: Pearson r and r², Spearman ρ, RMSE, each with bootstrap 95% confidence
intervals.

## Known data limitations

- **Label noise scales with read depth.** MRL is a read-weighted mean, so a UTR
  seen in few reads has a poorly estimated label. sd(error) is ~0.07 above
  1,000 reads against ~0.57 below 50. This caps achievable r².
- **MRL is clipped by construction** to [0, 13]; 429 real rows sit at exactly
  0.00 and 368 at exactly 13.00.
- **A 5'-end artefact affects ~3.9% of the library.** 6,324 near-duplicate
  groups differ by a single base, 81% of them at position 0 or 1, and the
  shallower member of each pair is ~10× less deeply sequenced than typical.
  These are grouped so they cannot straddle a split; they are not merged,
  because the evidence that they are the same oligo is strong but not
  conclusive.

## Ethical and safety considerations

Designed sequences are computational candidates only and require wet-lab
validation. Optimising hard against a learned model moves sequences away from
the distribution it was trained on, so predicted MRL for heavily optimised UTRs
should be treated as less reliable than for library-like sequences.

## Licence

MIT. See [LICENSE](LICENSE).
