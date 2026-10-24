"""Automatic transcription with PhoWhisper (PRICAI 2025, Sec. 4.1).

The paper verifies ASR output manually; the released protocol ships the verified
transcripts, so this module is only needed to apply the method to new data.
Requires the ``asr`` extra (``pip install -e ".[asr]"``).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from ..data.audio import SAMPLE_RATE, read_audio

log = logging.getLogger(__name__)


def transcribe(
    items,
    output_csv: str | Path,
    model: str = "vinai/PhoWhisper-large",
    device: str | None = None,
    batch_size: int = 8,
    language: str | None = None,
) -> Path:
    """Transcribe ``(utterance, locator)`` pairs into a CSV ``utterance,text``.

    Utterances already present in ``output_csv`` are skipped, so an interrupted run
    resumes where it stopped.
    """
    import torch
    from transformers import pipeline

    output_csv = Path(output_csv)
    done: set[str] = set()
    if output_csv.exists():
        with open(output_csv, newline="", encoding="utf-8") as handle:
            done = {row["utterance"] for row in csv.DictReader(handle)}
    todo = [(u, loc) for u, loc in items if u not in done]
    log.info("%d utterances already transcribed, %d to go", len(done), len(todo))
    if not todo:
        return output_csv

    cuda = torch.cuda.is_available() and device != "cpu"
    target = device or (0 if cuda else "cpu")
    dtype = torch.float16 if cuda else torch.float32
    try:
        asr = pipeline("automatic-speech-recognition", model=model, device=target, dtype=dtype, chunk_length_s=30)
    except TypeError:  # transformers < 4.56
        asr = pipeline("automatic-speech-recognition", model=model, device=target, torch_dtype=dtype, chunk_length_s=30)
    kwargs = {"generate_kwargs": {"language": language, "task": "transcribe"}} if language else {}

    new_file = not output_csv.exists()
    with open(output_csv, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["utterance", "text"])
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            audio = [{"raw": read_audio(loc), "sampling_rate": SAMPLE_RATE} for _, loc in batch]
            for (utterance, _), result in zip(batch, asr(audio, batch_size=batch_size, **kwargs)):  # type: ignore[arg-type, unused-ignore]
                writer.writerow([utterance, result["text"].strip()])
            handle.flush()
            log.info("transcribed %d/%d", len(done) + start + len(batch), len(done) + len(todo))
    return output_csv
