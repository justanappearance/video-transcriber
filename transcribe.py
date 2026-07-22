"""Point at a video file, get a full local transcript next to it.

Runs openai-whisper directly on GPU rather than ffmpeg's built-in whisper filter
(compare footage-triage/whisper_transcribe.py) - this needs full-length, high-fidelity
transcripts of long personal recordings, not quick keyword-spotting from short samples.
See DECISIONS.md for why, and for the GPU/CUDA setup gotcha on Pascal-era cards.
"""

import argparse
import subprocess
from pathlib import Path

DEFAULT_MODEL = "medium"


def transcribe(video_path: Path, model: str = DEFAULT_MODEL) -> Path:
    subprocess.run(
        [
            "whisper", str(video_path),
            "--model", model,
            "--device", "cuda",
            "--language", "English",
            "--output_format", "txt",
            "--output_dir", str(video_path.parent),
        ],
        check=True,
    )
    return video_path.with_suffix(".txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe a video locally on GPU.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    out = transcribe(args.video, args.model)
    print(f"Transcript written to {out}")
