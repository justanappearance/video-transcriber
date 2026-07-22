# video-transcriber

## What it does
A one-file CLI tool: point it at a video, get a full text transcript written next to it.
Built for Mike's habit of recording himself talking through his thoughts (e.g. working
through a dating situation) and wanting a full, high-fidelity, private transcript to paste
into a Claude chat afterward - not a summary, the raw rambling as-spoken, since the
hedging/repetition/self-correction is itself part of the signal he wants preserved.
Runs entirely locally on GPU, nothing leaves the machine.

## Architecture
Single script, `transcribe.py`. It's a thin `subprocess` wrapper around the `whisper` CLI
(from the `openai-whisper` PyPI package) - no whisper Python API calls, no batching, no
cleanup/summarization step. Cleanup, if wanted, is a separate manual pass Mike does himself
afterward so nothing gets silently rewritten.

## File map
- `transcribe.py` - the whole tool. `transcribe(video_path, model="medium")` runs
  `whisper <video> --device cuda --output_format txt --output_dir <same folder>`.

## Key decisions
See DECISIONS.md. Short version: `openai-whisper` + CUDA torch chosen over ffmpeg's
built-in whisper filter (used by the sibling `footage-triage` project) because this tool
needs full transcript fidelity, not fast keyword classification from short samples.

## Known gotchas
- **This machine's GPU (GTX 1060, Pascal, sm_61) needs the `cu126` CUDA build of torch,
  not newer ones.** `cu128`+ builds install fine and report
  `torch.cuda.is_available() == True`, but silently can't run real kernels on this card
  (warns "not compatible with the current PyTorch installation"). Always verify with an
  actual op, not just `is_available()`:
  ```
  python -c "import torch; print(torch.tensor([1.]).cuda() * 2)"
  ```
- No dedicated venv for this project (unlike most other Coding projects) - relies on the
  machine's global Python install. See DECISIONS.md for why.
- `whisper`'s CLI progress output is fully buffered when stdout isn't a real terminal
  (e.g. piped to a log file by a background task runner) - the log can look empty for
  the whole run even though it's actively working. Check `nvidia-smi` GPU utilization to
  confirm it's actually running rather than stuck.
- A 33-minute clip on this GPU with the `medium` model took a bit under 20 minutes.
