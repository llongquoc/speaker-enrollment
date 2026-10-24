# Checkpoints

The evaluation uses the speaker encoder of both papers:

| | |
|---|---|
| File | `ecapa_tdnn_vietnam_celeb.model` (62.7 MB) |
| SHA-256 | `af01bb9dcb87a42e820aa7679d6cc6d3617b3e8f10be40b10a05f1117bdd8149` |
| Architecture | ECAPA-TDNN, channels C = 1024, 80 log-Mel bands, 192-dimensional embeddings |
| Training data | Vietnam-Celeb training split; 900-way AAM-softmax layer whose classes match 893 corpus speakers ([details](../protocols/vietnam_celeb/README.md)) |
| Format | PyTorch state dict in the [TaoRuijie/ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN) layout (`speaker_encoder.*`, `speaker_loss.weight`) |

The file is stored with [Git LFS](https://git-lfs.com): with Git LFS installed (`git lfs install`),
`git clone` downloads it; in an existing clone, run `git lfs pull`.

Before computing any embedding, the pipeline checks the file against `model.sha256` in
[`configs/vietnam_celeb.yaml`](../configs/vietnam_celeb.yaml). To evaluate another encoder in the same
layout, set `model.checkpoint` to its path and `model.sha256` to `null`.

Other checkpoint files in this directory are ignored by git (see `.gitignore`).
