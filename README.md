# Enrollment Selection for Speaker Identification

[![Tests](https://github.com/llongquoc/speaker-enrollment/actions/workflows/tests.yml/badge.svg)](https://github.com/llongquoc/speaker-enrollment/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Official implementation of two papers by Long-Quoc Le and Minh-Nhut Ngo (Faculty of Information
Technology, University of Science, Vietnam National University, Ho Chi Minh City).

| Paper | Venue | Method |
|---|---|---|
| [Data-Centric Optimization of Enrollment Selection in Speaker Identification](https://doi.org/10.5220/0013256800003905) | ICPRAM 2025 | Enroll only mutually consistent utterances: every pair in the enrollment set reaches the decision threshold |
| [Phoneme-Based Optimization of Enrollment Selection for Speaker Identification](https://doi.org/10.1007/978-981-95-7075-1_34) | PRICAI 2025 | Among consistent utterances, prefer those that cover the most phonemes; generate phoneme-rich enrollment prompts |

The repository provides:

- **Selection methods** as a Python library: similarity-based selection (ICPRAM 2025) and
  phoneme-coverage-based selection (PRICAI 2025).
- **Deployment API**: quality-gated interactive enrollment, identification with rejection and
  phoneme-rich prompt generation.
- **Experiment pipeline**: runs the experiments of both papers on Vietnam-Celeb with one command, from
  audio to result tables and figures; the evaluation protocol, verified transcripts and checksums are
  included.

## Reproduced results

Speaker identification on the Vietnam-Celeb test set (57 speakers, 4,154 utterances), reproduced
with this repository. k is the number of enrollment utterances per speaker; F1 is macro-averaged over
the speakers.

| Metric | k | Bad | Random | Optimal | Low-P | High-P |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 1 | 0.747 | 0.904 | 0.926 | 0.853 | **0.947** |
|  | 3 | 0.879 | 0.954 | 0.965 | 0.939 | **0.973** |
|  | 5 | 0.895 | 0.966 | 0.973 | 0.964 | **0.977** |
|  | 10 | 0.913 | 0.973 | 0.977 | 0.976 | **0.979** |
| F1 | 1 | 0.697 | 0.882 | 0.906 | 0.834 | **0.926** |
|  | 3 | 0.875 | 0.942 | 0.954 | 0.925 | **0.964** |
|  | 5 | 0.906 | 0.955 | 0.964 | 0.956 | **0.966** |
|  | 10 | 0.931 | 0.957 | 0.963 | 0.964 | **0.967** |

| Configuration | Enrollment utterances | Role |
|---|---|---|
| Optimal | Every pair reaches the decision threshold | Method of ICPRAM 2025 |
| High-P | Optimal candidates with the highest phoneme coverage | Method of PRICAI 2025 |
| Random | Drawn at random | Baseline |
| Bad | Every pair falls below the decision threshold | Control |
| Low-P | Optimal candidates with the lowest phoneme coverage | Control |

Bad, Random and Optimal are averaged over 30 random draws; Low-P and High-P are deterministic.

<p align="center"><img src="docs/figures/pricai2025_metrics_vs_k.png" width="800" alt="Accuracy and F1 versus enrollment size"></p>

- Optimal outperforms Random at every k in accuracy, precision, recall and F1, and Random outperforms
  Bad.
- High-P has the highest accuracy at every k and the highest F1 at every k except k = 6 (0.962 against
  0.963 for Optimal).
- High-P matches Optimal at k = 10 with fewer enrollment utterances: 4 in accuracy (0.978 against
  0.977) and 3 in F1 (0.964 against 0.963).
- Selection matters more at stricter decision thresholds: at τ = 0.40 and k = 5, High-P reaches an
  accuracy of 0.923 and Optimal 0.913, against 0.863 for Random (F1 0.947, 0.942 and 0.910).

All metrics for k = 1–10, standard deviations and analyses: [docs/results.md](docs/results.md).

## Installation

Python 3.10 or newer and PyTorch 2.0 or newer.

```bash
git lfs install
git clone https://github.com/llongquoc/speaker-enrollment.git
cd speaker-enrollment
pip install -e .
pip install --no-deps viphoneme==3.0.0
```

- The speaker-encoder checkpoint is stored with [Git LFS](https://git-lfs.com); `git lfs install`
  (once per machine) makes `git clone` download it. In a clone made without Git LFS, run
  `git lfs pull`.
- `viphoneme` provides the Vietnamese syllable rules of the phoneme extractor. Only those rules are
  used, so it is installed without its text-normalisation dependencies.

## Data and model

The corpus is not part of the repository; download it once, from the repository root.

1. **Vietnam-Celeb corpus** (1,000 speakers, 87,140 utterances). Download the four parts listed in the
   [Vietnam-Celeb repository](https://github.com/Vietnam-Celeb/Vietnam-Celeb) into `data/`, then join
   them into one archive with the command given there:

   ```bash
   mkdir -p data
   cd data
   zip -F vietnam-celeb-part.zip --out full-dataset.zip
   cd ..
   ```

   The archive is read directly and does not need to be extracted. To use an extracted copy instead,
   set `corpus.path` in the configuration to its directory.

2. **ECAPA-TDNN checkpoint**: included as `checkpoints/ecapa_tdnn_vietnam_celeb.model` (Git LFS).

3. **Evaluation protocol**: included in `protocols/vietnam_celeb/`.

Expected layout:

```text
speaker-enrollment/
├── data/full-dataset.zip
├── checkpoints/ecapa_tdnn_vietnam_celeb.model
└── protocols/vietnam_celeb/
```

The checkpoint and the protocol files are checked against their SHA-256 checksums before use
([checkpoints/README.md](checkpoints/README.md), [SHA256SUMS](protocols/vietnam_celeb/SHA256SUMS)).

## Running the experiments

From the repository root:

```bash
speaker-enrollment reproduce --quick
speaker-enrollment reproduce
```

`--quick` is an end-to-end check on 6 test and 6 validation speakers (`outputs/vietnam_celeb_quick/`).
The full run writes `outputs/vietnam_celeb/REPORT.md` and takes a few minutes on a laptop: the 4,663
validation and test utterances are embedded once (cached; CUDA or Apple MPS is used when available),
then both experiments and the analyses run on CPU. Each step is also available as a command:

| Command | Step |
|---|---|
| `prepare` | Verify the protocol files, index the corpus, assign the test and validation subsets |
| `embed` | Extract and cache speaker embeddings |
| `calibrate` | Decision threshold at the validation equal-error-rate point |
| `icpram2025` | ICPRAM 2025 experiment: Bad, Random and Optimal enrollment for k = 1–10 |
| `phonemize` | Phoneme units and coverage of the test transcripts |
| `pricai2025` | PRICAI 2025 experiment: High-P and Low-P enrollment for k = 1–10 |
| `analyze` | Operating-point and design-choice analysis |
| `report` | Write the report; `--output docs/results.md` also copies its figures |
| `recover-split` | Derive the training and validation speakers from the checkpoint |
| `audit-labels` | Rank utterances by label consistency for manual review |
| `make-prompts` | Phoneme-rich enrollment prompts (PRICAI 2025, Algorithm 1) |
| `transcribe` | Transcribe new data with PhoWhisper (`pip install -e ".[asr]"`) |

All commands read [`configs/vietnam_celeb.yaml`](configs/vietnam_celeb.yaml); any value can be
overridden, for example `--set threshold.rule=fixed --set threshold.value=0.40`. The protocol and the
choice of every setting are documented in [docs/protocol.md](docs/protocol.md).

## Using the methods

```python
import numpy as np

from speaker_enrollment.data import read_audio
from speaker_enrollment.embedding import embed_waveform, load_checkpoint
from speaker_enrollment.enrollment import Enroller, SpeakerIdentifier
from speaker_enrollment.phonetics import make_g2p
from speaker_enrollment.selection import candidate_sets, select_by_coverage, select_enrollment

TAU = 0.1432  # decision threshold: validation equal-error-rate point of the released checkpoint
model, _ = load_checkpoint("checkpoints/ecapa_tdnn_vietnam_celeb.model")


def embed(path):
    return embed_waveform(model, read_audio(path))


# Select k = 5 enrollment utterances among a speaker's recordings
x = np.stack([embed(p) for p in paths])  # paths: recordings of one speaker
sim = x @ x.T
high, low = candidate_sets(sim, TAU)  # dominant similarity clique, the rest
optimal = select_enrollment(sim, 5, "optimal", TAU, np.random.default_rng(0), candidates=(high, low))
g2p = make_g2p("viphoneme")
coverage = np.array([g2p.coverage(text) for text in texts])  # texts: transcripts of the recordings
high_p = select_by_coverage(sim, coverage, 5, TAU, candidates=high)
print(optimal.indices, high_p.indices)

# Interactive enrollment: keep a recording only if it passes the similarity quality check
enroller = Enroller(threshold=TAU, target_size=5)
for path in new_recordings:  # e.g. readings of phoneme-rich prompts
    enroller.offer(embed(path))
    if enroller.complete:
        break
identifier = SpeakerIdentifier(threshold=TAU)
identifier.enroll("speaker-1", enroller.profile())
speaker, score = identifier.identify(embed("query.wav"))  # speaker is None if rejected
```

Phoneme-rich prompts from a text corpus (one word list per line, plus a coverage table):

```bash
speaker-enrollment make-prompts --corpus corpus.txt --sentences 10 --words 15 --output prompts.txt
```

## Repository structure

```text
configs/                  evaluation configuration
protocols/vietnam_celeb/  test list, trials, transcripts, speaker split, checksums
checkpoints/              speaker encoder (Git LFS)
docs/                     protocol and results
src/speaker_enrollment/
  data/                   corpus access (ZIP archive or directory), protocol readers
  embedding/              ECAPA-TDNN and embedding extraction
  selection/              similarity-based (ICPRAM 2025) and phoneme-based (PRICAI 2025) selection
  phonetics/              phoneme extraction, coverage, prompt generation, transcription
  experiments/            experiment runners, analyses, tables, figures, report
  verification.py         pairwise scoring and decision threshold
  identification.py       identification with rejection and its metrics
  enrollment.py           deployment-time enrollment and identification
  pipeline.py, cli.py     experiment pipeline and command-line interface
tests/                    unit and end-to-end tests
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests && ruff format --check src tests
mypy src
```

Optionally, `pre-commit install` runs the linters and the formatter on every commit.

## Citation

```bibtex
@inproceedings{le2025data,
  title={Data-Centric Optimization of Enrollment Selection in Speaker Identification.},
  author={Le, Long-Quoc and Ngo, Minh-Nhut},
  booktitle={ICPRAM},
  pages={344--351},
  year={2025}
}

@inproceedings{le2025correction,
  title={Correction to: Phoneme-Based Optimization of Enrollment Selection for Speaker Identification},
  author={Le, Long-Quoc and Ngo, Minh-Nhut},
  booktitle={Pacific Rim International Conference on Artificial Intelligence},
  pages={C1--C1},
  year={2025},
  organization={Springer}
}
```

## License

The code is released under the [MIT License](LICENSE). The Vietnam-Celeb corpus and models trained on
it are subject to the terms of the corpus.

## Acknowledgements

- Vietnam-Celeb corpus: Pham et al., Interspeech 2023.
- ECAPA-TDNN: Desplanques et al., Interspeech 2020; the encoder follows the layout of
  [TaoRuijie/ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN).
- Phoneme rules: [viphoneme](https://github.com/v-nhandt21/Viphoneme) (Nguyen et al., NICS 2020).
- Transcription: [PhoWhisper](https://huggingface.co/vinai/PhoWhisper-large) (Le et al., ICLR 2024 Tiny Papers).
