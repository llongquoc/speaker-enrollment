# Vietnam-Celeb evaluation protocol

Protocol files read by [`configs/vietnam_celeb.yaml`](../../configs/vietnam_celeb.yaml). Utterances are
identified by their path relative to the corpus audio directory, `<speaker>/<file>`
(for example `id00002/00000.wav`; the audio lies under `data/` in the Vietnam-Celeb distribution).

| File | Content |
|---|---|
| `test_utterances.txt` | 4,154 test utterances of 57 speakers, one per line |
| `test_trials.txt.gz` | 1,074,357 verification trials `label utterance_a utterance_b` (235,919 target, 838,438 non-target) |
| `test_transcripts.json` | Manually verified transcripts of the 4,154 test utterances (PRICAI 2025) |
| `test_speakers.txt` | The 57 test speakers |
| `validation_speakers.txt` | 50 speakers used to place the decision threshold |
| `training_speakers.txt` | 893 speakers of the encoder's training set |
| `SHA256SUMS` | Checksums of the files above |

Verify the files with `sha256sum -c SHA256SUMS` (Linux) or `shasum -a 256 -c SHA256SUMS` (macOS).

## Provenance

**Test set.** `test_utterances.txt` and `test_trials.txt.gz` come from the test trial list used in
both papers; the only change is that the leading `testout/` directory was removed from the paths.
The trial list contains every same-speaker pair of the test utterances. The transcripts were produced
with PhoWhisper and verified manually (PRICAI 2025, Sec. 4.1).

**Speaker split.** The corpus contains 1,000 speakers and 87,140 utterances. The training and
validation speaker lists were derived from the released checkpoint with

```bash
speaker-enrollment recover-split --write-protocol protocols/vietnam_celeb
```

The checkpoint's AAM-softmax layer has 900 classes. Each class is assigned to a distinct corpus
speaker by maximising the summed cosine between class weights and the speakers' mean embeddings
(Hungarian algorithm); assignments with a cosine below 0.25 are rejected as chance level. The
accepted and rejected assignments are far apart (lowest accepted 0.39, highest rejected 0.05):

| Role | Speakers | Utterances | Class match |
|---|---:|---:|---|
| Training | 893 | 79,794 | cosine ≥ 0.39 (median 0.89) |
| Test | 57 | 4,154 in the test list (6,837 in the corpus) | none |
| Validation | 50 | 509 | none |

Classes 893–899 match no corpus speaker, which indicates a 900-way classifier trained on 893
speakers. The validation speakers are the remaining speakers: absent from training and disjoint from
the test speakers, as the protocol requires.
