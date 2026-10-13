"""Point at a video file, get a full local transcript in a dedicated transcripts folder.

Uses faster-whisper (CTranslate2 backend) directly, not ffmpeg's built-in whisper filter
(compare footage-triage/whisper_transcribe.py) - this needs full-length, high-fidelity
transcripts of long personal recordings, not quick keyword-spotting from short samples.
See DECISIONS.md for why, and for the GPU compute-type gotcha on Pascal-era cards.
"""

import argparse
import os
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from faster_whisper import WhisperModel, BatchedInferencePipeline

# Caption cues for the .srt are capped at these sizes (word-level timestamps, not Whisper's
# own segments - a segment can run 20-30s on Mike's rambling-talk recordings, too long to
# read as an on-screen caption). Numbers match CapCut's auto-captions (the social-caption
# style Mike's going for) by observation - CapCut doesn't publish a spec - so treat these as
# a starting point to adjust, not a fixed target. The .txt output is unaffected, still one
# line per Whisper segment.
MAX_CAPTION_CHARS = 90
MAX_CAPTION_LINE_CHARS = 38

DEFAULT_MODEL = "medium"
# 16 OOM'd on a 31-min video (6GB Pascal card) - BatchedInferencePipeline pads every
# segment in a batch to the longest one, so a batch containing a couple of unusually long
# VAD segments can spike memory well past what the 17-min test clip showed. See DECISIONS.md.
BATCH_SIZE = 4

# Lives inside the repo now (Mike consolidated everything under C:\Coding), but this repo
# is public on GitHub - transcripts are personal and must never be committed. Protected via
# .gitignore (transcripts/), not by keeping the folder outside the repo. See DECISIONS.md.
TRANSCRIPTS_DIR = Path(r"C:\Coding\video-transcriber\transcripts")


def _recording_timestamp(video_path: Path) -> float:
    """Phone videos embed a real creation_time tag - use that for the output files'
    mtimes instead of leaving them at "whenever transcribe.py happened to run". Some
    journals now auto-transcribe the instant they land, others sit until Mike asks for
    them by hand days later, which used to scramble the transcripts folder's Date
    Modified order relative to actual recording order. Falls back to the video file's
    own mtime if the tag is missing (e.g. a re-encoded or non-phone source)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if out:
            return datetime.strptime(out, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        pass
    return video_path.stat().st_mtime


def _srt_timestamp(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _chunk_words_into_captions(words, max_chars: int = MAX_CAPTION_CHARS):
    """Groups word-level timestamps into caption-sized chunks, each capped at `max_chars`
    characters of text - starts a new chunk rather than let one run longer, so a single
    very long word still gets its own chunk instead of being split mid-word."""
    chunks = []
    current = []
    current_chars = 0
    for word in words:
        if current and current_chars + len(word.word) > max_chars:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(word)
        current_chars += len(word.word)
    if current:
        chunks.append(current)
    return chunks


def transcribe(video_path: Path, model: str = DEFAULT_MODEL, output_dir: Path = TRANSCRIPTS_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    whisper_model = WhisperModel(model, device="cuda", compute_type="int8")
    batched_model = BatchedInferencePipeline(model=whisper_model)
    segments, _ = batched_model.transcribe(
        str(video_path), language="en", batch_size=BATCH_SIZE, word_timestamps=True
    )

    out_path = output_dir / (video_path.stem + ".txt")
    srt_path = output_dir / (video_path.stem + ".srt")
    all_words = []
    with out_path.open("w", encoding="utf-8") as txt_f:
        for segment in segments:
            txt_f.write(segment.text.strip() + "\n")
            all_words.extend(segment.words)

    with srt_path.open("w", encoding="utf-8") as srt_f:
        for i, chunk in enumerate(_chunk_words_into_captions(all_words), start=1):
            text = "".join(w.word for w in chunk).strip()
            lines = "\n".join(textwrap.wrap(text, width=MAX_CAPTION_LINE_CHARS))
            start, end = chunk[0].start, chunk[-1].end
            srt_f.write(f"{i}\n{_srt_timestamp(start)} --> {_srt_timestamp(end)}\n{lines}\n\n")

    recorded_at = _recording_timestamp(video_path)
    os.utime(out_path, (recorded_at, recorded_at))
    os.utime(srt_path, (recorded_at, recorded_at))
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe a video locally on GPU.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=TRANSCRIPTS_DIR)
    args = parser.parse_args()
    out = transcribe(args.video, args.model, args.output_dir)
    print(f"Transcript written to {out}")
