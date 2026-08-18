"""Point at a video file, get a full local transcript in a dedicated transcripts folder.

Uses faster-whisper (CTranslate2 backend) directly, not ffmpeg's built-in whisper filter
(compare footage-triage/whisper_transcribe.py) - this needs full-length, high-fidelity
transcripts of long personal recordings, not quick keyword-spotting from short samples.
See DECISIONS.md for why, and for the GPU compute-type gotcha on Pascal-era cards.
"""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from faster_whisper import WhisperModel, BatchedInferencePipeline

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


def transcribe(video_path: Path, model: str = DEFAULT_MODEL, output_dir: Path = TRANSCRIPTS_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    whisper_model = WhisperModel(model, device="cuda", compute_type="int8")
    batched_model = BatchedInferencePipeline(model=whisper_model)
    segments, _ = batched_model.transcribe(str(video_path), language="en", batch_size=BATCH_SIZE)

    out_path = output_dir / (video_path.stem + ".txt")
    # Also save timestamps alongside the plain-text transcript - added for the
    # video-journal-auto-cut project, which needs to align transcript segments back to
    # specific points in the raw footage. The plain .txt output is untouched (still the
    # thing Mike pastes into a chat), this is purely additive.
    json_path = output_dir / (video_path.stem + ".json")
    segments = list(segments)  # segments is a generator - only consumable once
    with out_path.open("w", encoding="utf-8") as f:
        for segment in segments:
            f.write(segment.text.strip() + "\n")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments],
            f, indent=2,
        )

    recorded_at = _recording_timestamp(video_path)
    os.utime(out_path, (recorded_at, recorded_at))
    os.utime(json_path, (recorded_at, recorded_at))
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe a video locally on GPU.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=TRANSCRIPTS_DIR)
    args = parser.parse_args()
    out = transcribe(args.video, args.model, args.output_dir)
    print(f"Transcript written to {out}")
