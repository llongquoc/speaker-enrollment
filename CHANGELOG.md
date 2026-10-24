# Changelog

## 1.0.0 — 2026-10-24

First public release: the official implementation of the ICPRAM 2025 and PRICAI 2025 papers.

- Similarity-based enrollment selection (ICPRAM 2025): Bad, Random and Optimal configurations with
  the dominant-clique candidate rule (alternatives: `mean`, `all`).
- Phoneme-based enrollment selection (PRICAI 2025): coverage-ordered High-P and Low-P selection,
  viphoneme-based phoneme extraction with the compound-vowel refinement, a dependency-free
  rule-based extractor, phoneme-rich prompt generation (Algorithm 1) and PhoWhisper transcription.
- ECAPA-TDNN encoder compatible with TaoRuijie/ECAPA-TDNN checkpoints; whole-utterance embeddings on
  CPU, CUDA or Apple MPS, with an agreement check against CPU.
- Identification with rejection; accuracy and macro-averaged precision, recall and F1.
- Experiment pipeline and command-line interface: corpus access from a ZIP archive or a directory,
  checksum verification of the checkpoint and the protocol files, embedding cache, threshold
  calibration (`eer`, `far`, `fixed`), both experiments, operating-point and design-choice analyses,
  figures, and a Markdown report that can be written to the documentation.
- ECAPA-TDNN checkpoint trained on Vietnam-Celeb, stored with Git LFS.
- Vietnam-Celeb protocol: test list, trial list, verified transcripts, and the training and validation
  speaker lists recovered from the checkpoint.
- Deployment API: quality-gated `Enroller` and `SpeakerIdentifier`.
- Unit and end-to-end tests, ruff and mypy configuration, pre-commit hooks; continuous integration runs
  the tests and ruff on Python 3.10, 3.12 and 3.13.
