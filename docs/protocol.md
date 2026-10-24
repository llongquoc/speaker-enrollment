# Evaluation protocol

This document specifies the evaluation run by `speaker-enrollment reproduce` and the reasons for each
setting. Reproduced results: [results.md](results.md).

## Pipeline

| Step | Command | Output (in `output_dir`) |
|---|---|---|
| 1. Manifest | `prepare` | `manifest.csv`: every corpus utterance with its subset (`test`, `validation`, `other`) |
| 2. Embeddings | `embed` | `embeddings/{validation,test}.npz`, cached and recomputed only when the utterances, their labels or the checkpoint change |
| 3. Threshold | `calibrate` | `threshold.json`: τ, validation FAR and FRR, test EER |
| 4. ICPRAM 2025 | `icpram2025` | `icpram2025/`: per-draw results, summary, figures |
| 5. Phonemes | `phonemize` | `phonemes/test_phonemes.csv`: phoneme units and coverage of each test utterance |
| 6. PRICAI 2025 | `pricai2025` | `pricai2025/`: results, summary, figures |
| 7. Analyses | `analyze` | `analysis/`: operating-point sweep, design choices, figure |
| 8. Report | `report` | `REPORT.md`; `run_info.json` (versions, checkpoint hash, configuration); `run.log` |

Each step rewrites its own output directory, so results from earlier runs never mix with new ones.

## Data

| Set | Speakers | Utterances | Use |
|---|---:|---:|---|
| Corpus (Vietnam-Celeb) | 1,000 | 87,140 | read from the distribution archive or an extracted copy |
| Test | 57 | 4,154 (13–315 per speaker, median 57) | enrollment and queries |
| Validation | 50 | 509 | decision threshold only |
| Training | 893 | 79,794 | not used by the evaluation |

- The test utterances, trials and transcripts form the released test protocol
  ([protocols/vietnam_celeb](../protocols/vietnam_celeb/README.md)); `prepare` checks them against
  `SHA256SUMS`.
- The validation speakers are absent from training and from the test set
  ([provenance](../protocols/vietnam_celeb/README.md#provenance)).
- Labels are the corpus speaker directories. Corrected labels can be supplied with `protocol.relabel`
  (CSV `utterance,speaker`; the speaker `DROP` removes an utterance).

## Speaker embeddings

- ECAPA-TDNN checkpoint ([checkpoints](../checkpoints/README.md)), checked against `model.sha256`.
- 16 kHz mono audio. Each whole utterance is embedded in its own forward pass: the encoder has no
  length masking, so neither cropping nor padding to a common length is applied. Embeddings are
  192-dimensional and L2-normalised.
- `model.device: auto` uses CUDA, then Apple MPS, then CPU. A GPU is used only if its embedding of a
  reference utterance agrees with the CPU embedding (cosine ≥ 0.999).

## Decision threshold

- Two utterances are attributed to the same speaker when their cosine similarity is at least τ.
- τ is the equal-error-rate point over all utterance pairs of the validation speakers (4,669 target
  and 124,617 non-target pairs), as in both papers: **τ = 0.1432** (EER 11.29%).
- The test EER is reported for reference only (4.53% over all test pairs, 4.54% over the trial list).

| `threshold.rule` | τ |
|---|---|
| `eer` (default) | validation equal-error-rate point |
| `far` | lowest τ with validation FAR ≤ `threshold.target_far` |
| `fixed` | `threshold.value` |

## Identification

For each configuration and enrollment size k, an enrollment set is selected for every test speaker;
all remaining test utterances are queries (closed set of 57 speakers).

| Element | Definition |
|---|---|
| Speaker score | mean cosine between the query and the speaker's enrollment utterances |
| Decision | best-scoring speaker; rejected as unknown if its score is below τ |
| Accuracy | correct decisions / queries; a rejection counts as an error |
| Precision, recall, F1 | per enrolled speaker, macro-averaged; unknown is not a class, so a rejection lowers recall but not precision; empty denominators count as 0 |
| Rejection rate | rejected queries / queries |

## ICPRAM 2025: similarity-based selection

Per test speaker, the cosine similarities between all its utterances are computed once.

| Configuration | Enrollment set of size k |
|---|---|
| Optimal (proposed) | drawn from the high-quality candidates; every pair has cos ≥ τ |
| Random (baseline) | drawn uniformly from all utterances |
| Bad (control) | drawn from the low-quality candidates; every pair has cos < τ |

- **Candidates.** At k = 1 no pair exists, so the configurations are also distinguished by where
  utterances are drawn from. The high-quality candidates of a speaker are the dominant clique of the
  graph that joins utterance pairs with cos ≥ τ (the high-similarity cluster of ICPRAM 2025, Fig. 2),
  found greedily and maximal; the other utterances are the low-quality candidates.
- **Sampling.** A random permutation of the candidates is scanned, and an utterance is accepted if it
  satisfies the pairwise constraint with every accepted one (up to 50 restarts).
- **Fallback.** If no restart reaches k utterances, the largest set found is completed with the
  utterances that violate the constraint least (Optimal: highest minimum similarity to the set; Bad:
  lowest maximum similarity); the count is recorded in the `fallback` column. At τ = 0.1432 nearly
  all pairs of a speaker's utterances reach τ, so most Bad-case utterances are chosen this way; the
  Optimal case never needs it.
- **Repetitions.** 30 draws per configuration and k, seeded by (seed, configuration, k, draw); mean
  and population standard deviation are reported.

## PRICAI 2025: phoneme-based selection

| Element | Definition |
|---|---|
| Transcripts | PhoWhisper output, verified manually (released with the protocol) |
| Phoneme units | viphoneme 3.0.0 syllable rules (Northern dialect, six tones) with the PRICAI 2025 refinement: a nucleus followed by an off-glide coda is one compound-vowel unit (*câu* → k, ɤ̆ʷ); tones are separate units T1–T6 |
| Coverage of an utterance | number of distinct units in its transcript; tokens that are not Vietnamese syllables add no units (720 of the 4,154 transcripts contain some: loanwords, foreign names, single letters, the placeholder `unk`) |
| P of an enrollment set | number of distinct units in the union of its transcripts, averaged over speakers |
| High-P (proposed) | the speaker's high-quality candidates in descending order of coverage; an utterance is accepted if cos ≥ τ with every utterance already accepted, until k are accepted |
| Low-P (control) | the same in ascending order of coverage |
| Baseline, Similarity | the Random and Optimal configurations of the ICPRAM 2025 experiment, under the names used in PRICAI 2025 |

Selection is deterministic (one run per k). The extractor yields 93 distinct units over the test
transcripts.

## Choice of settings

The settings that the method descriptions leave open were compared by `analyze`
([results](results.md#design-choices)).

| Setting | Value | Reason |
|---|---|---|
| Candidate rule | `clique` | Highest mean Optimal and High-P accuracy and F1 among the candidate rules, for every scoring rule (`mean`: mean similarity to the speaker's other utterances ≥ τ; `all`: no split) |
| Speaker scoring | `mean` | Mean of the pairwise cosines, on the same scale as τ, so a query is accepted under the criterion calibrated on the validation pairs; with `clique` candidates, Optimal > Random > Bad and High-P > Low-P hold at every k in accuracy and F1. Centroid scores give a higher mean accuracy and F1 but exceed pairwise cosines, so τ loses its calibrated meaning, and Optimal > Random > Bad fails for k ≥ 6; with maximum scores it fails for some k ≥ 6 |
| Quality check | `pairwise` | As in ICPRAM 2025; with `clique` candidates every pair already reaches τ, so the `mean` check gives identical results |
| Threshold rule | `eer` | Validation equal-error-rate point, as in both papers; τ is fixed before any test data is used |

## Analyses

| Analysis | Setting |
|---|---|
| Operating point | Both experiments at τ = 0.10–0.50 in steps of 0.02 and at the calibrated τ (10 random draws per configuration), with FAR and FRR over all validation and test utterance pairs |
| Design choices | Every combination of candidate rule, scoring rule and quality check at the calibrated τ (10 random draws per configuration) |

## Configuration

All settings are in [`configs/vietnam_celeb.yaml`](../configs/vietnam_celeb.yaml). Any value can be
overridden on the command line; `--reuse-embeddings` copies matching embeddings from an earlier run
instead of recomputing them:

```bash
speaker-enrollment reproduce --set threshold.rule=fixed --set threshold.value=0.40 \
    --output-dir outputs/tau_0.40 --reuse-embeddings outputs/vietnam_celeb
speaker-enrollment reproduce --set identification.scoring=centroid \
    --output-dir outputs/centroid --reuse-embeddings outputs/vietnam_celeb
```
