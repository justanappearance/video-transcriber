# video-transcriber

## What it does
A one-file CLI tool: point it at a video, get a full text transcript in
`transcripts/` inside this repo (gitignored - see DECISIONS.md for why that's safe despite
this repo being public on GitHub). Built for Mike's habit of recording himself talking
through his thoughts (e.g. working through a dating situation) and wanting a full,
high-fidelity, private transcript to paste into a Claude chat afterward - not a summary,
the raw rambling as-spoken, since the hedging/repetition/self-correction is itself part of
the signal he wants preserved. Runs entirely locally on GPU, nothing leaves the machine.

## Architecture
Single script, `transcribe.py`. It calls the `faster-whisper` Python API directly, wrapping
`WhisperModel` in a `BatchedInferencePipeline` for GPU throughput - no subprocess, no
cleanup/summarization step. Cleanup, if wanted, is a separate manual pass Mike does himself
afterward so nothing gets silently rewritten.

## File map
- `transcribe.py` - the whole tool. `transcribe(video_path, model="medium", output_dir=TRANSCRIPTS_DIR)`
  builds `WhisperModel(model, device="cuda", compute_type="int8")`, wraps it in
  `BatchedInferencePipeline`, and calls `.transcribe(..., batch_size=16)`, writing one
  segment's text per line to `<output_dir>/<video stem>.txt`. `output_dir` defaults to
  `C:\Coding\video-transcriber\transcripts` but can be overridden with `--output-dir`.

## Key decisions
See DECISIONS.md. Short version: full transcript fidelity (not fast keyword
classification from short samples, which is what the sibling `footage-triage` project
needs) is why this uses a real Whisper model end-to-end rather than ffmpeg's built-in
whisper filter. Originally built on the `openai-whisper` CLI + CUDA torch; switched to
`faster-whisper` (CTranslate2 backend) once measured ~4x faster on this GPU for the same
quality - see DECISIONS.md for the switch and why torch wasn't tried first.

## Known gotchas
- **`transcripts/` is gitignored, not absent.** The folder lives inside this public repo's
  checkout now (`C:\Coding\video-transcriber\transcripts`) - it's protected from ever being
  committed by `.gitignore`, not by living outside the repo like it used to. Don't remove
  that gitignore entry without re-reading DECISIONS.md.
- **This GPU (GTX 1060, Pascal, sm_61) doesn't support `int8_float16` compute type in
  CTranslate2** - fails with "Requested int8_float16 compute type, but the target device
  or backend do not support efficient int8_float16 computation." Plain `int8` works fine
  and is what `transcribe.py` uses.
- No dedicated venv for this project (unlike most other Coding projects) - relies on the
  machine's global Python install. See DECISIONS.md for why (written when the dependency
  was still torch-based and much heavier; `faster-whisper` alone is light enough that a
  venv would be cheap too, but there was no reason to add one once things worked).
- A 17-minute clip on this GPU with the `medium` model + `int8` took ~90 seconds (vs.
  ~5-6 minutes with the old `openai-whisper` CLI path). Since switching to
  `BatchedInferencePipeline` (batch_size=16), it's faster still - see DECISIONS.md.
