"""Point at a video file, get a full local transcript in a dedicated transcripts folder.

Uses faster-whisper (CTranslate2 backend) directly, not ffmpeg's built-in whisper filter
(compare footage-triage/whisper_transcribe.py) - this needs full-length, high-fidelity
transcripts of long personal recordings, not quick keyword-spotting from short samples.
See DECISIONS.md for why, and for the GPU compute-type gotcha on Pascal-era cards.
"""

import argparse
from pathlib import Path

from faster_whisper import WhisperModel

DEFAULT_MODEL = "medium"

# Deliberately outside this repo (which is public) and outside C:\Videos entirely - a
# transcript is Claude sounding-board material, not a video asset, and footage\ contents
# get moved/recategorized/deleted by the footage-triage pipeline. See DECISIONS.md.
TRANSCRIPTS_DIR = Path(r"C:\Programs\video-transcriber\transcripts")


def transcribe(video_path: Path, model: str = DEFAULT_MODEL, output_dir: Path = TRANSCRIPTS_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    whisper_model = WhisperModel(model, device="cuda", compute_type="int8")
    segments, _ = whisper_model.transcribe(str(video_path), language="en")
    out_path = output_dir / (video_path.stem + ".txt")
    with out_path.open("w", encoding="utf-8") as f:
        for segment in segments:
            f.write(segment.text.strip() + "\n")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe a video locally on GPU.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=TRANSCRIPTS_DIR)
    args = parser.parse_args()
    out = transcribe(args.video, args.model, args.output_dir)
    print(f"Transcript written to {out}")
