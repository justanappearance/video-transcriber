"""Point at a video file, get a full local transcript in a dedicated transcripts folder.

Uses faster-whisper (CTranslate2 backend) directly, not ffmpeg's built-in whisper filter
(compare footage-triage/whisper_transcribe.py) - this needs full-length, high-fidelity
transcripts of long personal recordings, not quick keyword-spotting from short samples.
See DECISIONS.md for why, and for the GPU compute-type gotcha on Pascal-era cards.
"""

import argparse
import json
import os
import re
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from faster_whisper import WhisperModel, BatchedInferencePipeline

# Caption cues for the .srt are capped at these sizes (word-level timestamps, not Whisper's
# own segments - a segment can run 20-30s on Mike's rambling-talk recordings, too long to
# read as an on-screen caption). Started at CapCut's observed 90/38 (its own auto-caption
# sizing), but at the large font size Mike's vertical-video subtitle style uses, 90 chars
# wrapped into 2-3 stacked lines read as "a lot of text on screen" (his words, 2026-10-12,
# after seeing it live in Resolve) - cut down to a single short punchy line instead. Treat
# these as a starting point to adjust, not a fixed target. The .txt output is unaffected,
# still one line per Whisper segment.
MAX_CAPTION_CHARS = 24
MAX_CAPTION_LINE_CHARS = 24

# A cue's on-screen time is the longer of (a) how long it actually took Mike to say it and
# (b) how long it takes to read, at READING_CHARS_PER_SECOND - fast speech on a long cue
# would otherwise flash by unreadably. Netflix's English subtitle style guide (the most
# commonly cited public reading-speed standard) caps adult content at 20 CPS and sets a
# 5/6-second minimum per cue - used here as-is, not re-derived.
READING_CHARS_PER_SECOND = 20
MIN_CAPTION_SECONDS = 5 / 6

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


_PRONOUN_I = re.compile(r"^i(?:'(?:m|ve|ll|d))?$", re.IGNORECASE)


def _is_phrase_start(word_text: str) -> bool:
    """Whisper mostly drops sentence-ending punctuation on Mike's rambling recordings but
    still capitalizes the first word of each new phrase - the only boundary signal left.
    "I"/"I'm"/"I've"/etc. are excluded since they're capitalized anywhere in a sentence."""
    stripped = word_text.strip()
    return bool(stripped) and stripped[0].isupper() and not _PRONOUN_I.match(stripped)


def _merge_split_words(words):
    """Whisper sometimes emits one word as several tokens (e.g. " self" + "-control"). Every
    real word starts with a space, so a token without one is a continuation of the previous
    word - merge it back so a cue can never end mid-word."""
    merged = []
    for word in words:
        if merged and not word.word[:1].isspace():
            prev = merged[-1]
            merged[-1] = SimpleNamespace(word=prev.word + word.word, start=prev.start, end=word.end)
        else:
            merged.append(SimpleNamespace(word=word.word, start=word.start, end=word.end))
    return merged


def _chunk_words_into_captions(words, max_chars: int = MAX_CAPTION_CHARS):
    """Groups word-level timestamps into caption cues. A cue ends before any word that
    starts a new phrase (so the tail of one sentence never shares a cue with the start of
    the next), or before a word that would push it past `max_chars` - the backstop for a
    long run-on phrase. Never splits mid-word."""
    chunks = []
    current = []
    current_chars = 0
    for word in words:
        starts_new_phrase = _is_phrase_start(word.word)
        over_budget = current_chars + len(word.word) > max_chars
        if current and (starts_new_phrase or over_budget):
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(word)
        current_chars += len(word.word)
    if current:
        chunks.append(current)
    return chunks


def _build_caption_cues(words):
    """Chunks words into cues and gives each its on-screen time: the longer of the spoken
    duration and the reading-speed floor, but never running into the next cue."""
    chunks = _chunk_words_into_captions(words)
    cues = []
    for i, chunk in enumerate(chunks):
        text = "".join(w.word for w in chunk).strip()
        start = chunk[0].start
        spoken_end = chunk[-1].end
        readable_end = start + max(MIN_CAPTION_SECONDS, len(text) / READING_CHARS_PER_SECOND)
        end = max(spoken_end, readable_end)
        if i + 1 < len(chunks):
            end = max(spoken_end, min(end, chunks[i + 1][0].start))
        cues.append(SimpleNamespace(text=text, start=start, end=end, words=chunk))
    return cues


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

    cues = _build_caption_cues(_merge_split_words(all_words))
    with srt_path.open("w", encoding="utf-8") as srt_f:
        for i, cue in enumerate(cues, start=1):
            lines = "\n".join(textwrap.wrap(cue.text, width=MAX_CAPTION_LINE_CHARS))
            srt_f.write(f"{i}\n{_srt_timestamp(cue.start)} --> {_srt_timestamp(cue.end)}\n{lines}\n\n")

    # Same cues as the .srt, plus each word's own timing - render_captions.py needs those to
    # move the highlight box word by word, which an .srt has no way to carry.
    captions_path = output_dir / (video_path.stem + ".captions.json")
    captions_path.write_text(json.dumps([
        {"start": cue.start, "end": cue.end,
         "words": [{"text": w.word.strip(), "start": w.start, "end": w.end} for w in cue.words]}
        for cue in cues
    ], indent=2), encoding="utf-8")

    recorded_at = _recording_timestamp(video_path)
    for path in (out_path, srt_path, captions_path):
        os.utime(path, (recorded_at, recorded_at))
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe a video locally on GPU.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=TRANSCRIPTS_DIR)
    args = parser.parse_args()
    out = transcribe(args.video, args.model, args.output_dir)
    print(f"Transcript written to {out}")
