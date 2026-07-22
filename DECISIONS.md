## openai-whisper + CUDA torch, not ffmpeg's built-in whisper filter
- Decision: transcribe with the `openai-whisper` PyPI package (`medium` model, GPU) via
  its `whisper` CLI, rather than ffmpeg's `-af whisper` filter used by the sibling
  `footage-triage` project's `whisper_transcribe.py`.
- Why: this tool exists to produce full, high-fidelity transcripts of 30+ minute personal
  talk-throughs for feeding into a Claude chat - quality and completeness matter more than
  dependency weight. footage-triage's use case is the opposite: a quick category guess from
  three 60-second samples through a small `base.en` model, where footage-triage explicitly
  chose the ffmpeg filter specifically to *avoid* adding `openai-whisper`/`faster-whisper`
  as a dependency (numba/CTranslate2 wheel-availability risk on a very new Python).
- Rejected: ffmpeg's built-in whisper filter (same approach as footage-triage).
- Why rejected: `base.en` quality plus short sampled windows would lose most of the
  rambling/nuance this tool exists to preserve.
- Note: footage-triage's dependency-risk concern was real - see the next entry - but it
  was worth accepting here given the quality requirement.
- Date: 2026-07-21

## Pascal-era GPU needs the cu126 torch build specifically
- Decision: install torch via
  `pip install torch --index-url https://download.pytorch.org/whl/cu126 --force-reinstall`.
- Why: this machine's GTX 1060 is Pascal (compute capability 6.1 / sm_61). The default/latest
  CUDA wheel index (`cu128`) installs a torch build that reports
  `torch.cuda.is_available() == True` but can't actually run kernels on this card - it warns
  "not compatible with the current PyTorch installation" the moment a real op is attempted.
  `cu126` (torch 2.13.0+cu126) still targets sm_61 and was verified with a real
  `.cuda()` tensor op.
- Rejected: `cu128` (torch 2.11.0+cu128) - installs, `is_available()` lies; `cu121`/`cu124`/
  `cu129` - no wheel exists at all for this Python version (cp314) as of this writing.
- Why rejected: incompatible with either this GPU or this Python version.
- Date: 2026-07-21

## No dedicated venv for this project
- Decision: relies on the machine's global Python install rather than a per-project venv
  (unlike most other Coding projects, e.g. footage-triage's `venv/`).
- Why: torch + CUDA is a multi-GB dependency. Duplicating it into an isolated venv costs
  several GB of disk and a long reinstall, for isolation benefit that doesn't matter on a
  personal single-user machine where nothing else conflicts with this exact dependency.
- Rejected: per-project venv, matching repo convention.
- Why rejected: cost not worth it here.
- Date: 2026-07-21

## Switched from openai-whisper to faster-whisper
- Decision: `transcribe.py` now calls `faster-whisper` (CTranslate2 backend) directly via
  its Python API, `compute_type="int8"`, instead of shelling out to the `openai-whisper`
  CLI. `requirements.txt` is now just `faster-whisper` - torch is no longer a dependency
  of this project at all.
- Why: Mike asked if transcription could be faster. Measured directly on this GPU with the
  same `medium` model: a 17-minute clip took ~90 seconds with `faster-whisper`/int8 vs.
  ~5-6 minutes with the `openai-whisper` CLI path - roughly a 4x speedup for equivalent
  quality, and no torch/CUDA-index dependency headache (see the two entries above) at all.
- Why `openai-whisper` was picked first instead of this: the original setup carried over
  `footage-triage`'s stated reason for avoiding `faster-whisper`/`ctranslate2` (numba/
  wheel-availability risk on this Python version) without re-checking whether that still
  held here. It didn't - `pip install faster-whisper` resolves a normal wheel (`ctranslate2`
  4.8.1) for this Python version with no issue. Lesson: verify an inherited decision's
  premise before reusing it in a new project, don't just carry the caution forward.
- Gotcha hit during the switch: `compute_type="int8_float16"` fails outright on this GPU
  (Pascal/sm_61) - "Requested int8_float16 compute type, but the target device or backend
  do not support efficient int8_float16 computation." Plain `int8` works.
- Rejected: staying on `openai-whisper`; also tried `int8_float16` before falling back to
  `int8`.
- Why rejected: openai-whisper measured ~4x slower for the same output quality; int8_float16
  isn't supported on this GPU at all.
- Date: 2026-07-21
