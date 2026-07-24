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

## Transcripts write to a dedicated C:\Videos\_transcripts folder, not next to the source video
- Decision: `transcribe.py` now writes output to `C:\Videos\_transcripts\<video stem>.txt`
  by default (overridable via `--output-dir`), instead of next to the source video file.
- Why: source videos live under `C:\Videos\footage\...` and get moved/recategorized/deleted
  by the sibling `footage-triage` project as part of its normal operation. Writing the
  transcript next to the video ties the transcript's lifecycle to the video's - concretely
  broke once already: a transcript ended up stranded in a `_old_camera_folder_debris` folder
  after its source video was triaged elsewhere, and was later cleaned up entirely by the
  pipeline before anyone moved it out. A dedicated folder outside `footage\` isn't touched
  by that pipeline at all.
- Also why it's `C:\Videos\_transcripts` and not somewhere under this repo: this repo is
  public on GitHub; transcript content (personal, sometimes about specific people) must
  never be anywhere near a folder that gets `git add`ed. Naming matches the existing
  underscore-prefixed utility-folder convention under `C:\Videos` (`_inbox`,
  `_google_drive_review`, `_youtube_review`).
- Rejected: leaving transcripts next to the source video (status quo); putting them inside
  this repo for convenience.
- Why rejected: former already caused a lost transcript; latter risks personal content
  landing in a public repo.
- Date: 2026-07-21
- Superseded 2026-07-21: see next entry - Mike moved this again the same day, `C:\Videos`
  turned out to be the wrong tree entirely, not just the wrong subfolder.

## Moved again: transcripts live at C:\Programs\video-transcriber\transcripts, not under C:\Videos at all
- Decision: `TRANSCRIPTS_DIR` is now `C:\Programs\video-transcriber\transcripts`. Note this
  is a *different folder* from this repo's own checkout (`...\Coding\video-transcriber\`) -
  same base name, coincidentally, but one is code (this repo, public) and the other is
  private output data. Don't confuse the two when looking for either.
- Why: Mike's framing - `C:\Videos` is for video assets; a transcript of a video journal
  isn't really that, it's Claude sounding-board material he's going to paste into a chat.
  The previous entry only fixed *which pipeline could disturb the file*, not the deeper
  mismatch of storing conversation-bound text under a folder tree that's conceptually about
  video files.
- Rejected: `C:\Videos\_transcripts` (previous entry).
- Why rejected: right instinct (decouple from footage-triage churn), wrong tree - still
  filed under "videos" a folder of content whose actual purpose is text input to Claude.
- Date: 2026-07-21
- Superseded 2026-07-24: see next entry - Mike moved his whole Coding folder to `C:\Coding`
  and wants transcripts consolidated inside the repo checkout itself.

## Moved again: transcripts live inside the repo at C:\Coding\video-transcriber\transcripts, protected by .gitignore
- Decision: `TRANSCRIPTS_DIR` is now `C:\Coding\video-transcriber\transcripts`, inside this
  repo's own checkout. Added `transcripts/` to `.gitignore` so the folder's contents are
  never tracked or committed.
- Why: Mike moved his entire Coding folder from OneDrive to `C:\Coding` and, in the process,
  had already manually consolidated the existing transcript files into the repo's new
  location himself before asking me to formalize it. Rather than fight that and put them
  back outside the repo, the safety property that mattered (personal content never reaching
  a public GitHub push) is preserved by gitignoring the folder instead of relocating it
  outside the repo tree.
- Rejected: keeping transcripts outside any repo folder (previous entry, and a sibling
  option floated - `C:\Coding\_transcripts` next to the repo rather than inside it).
- Why rejected: Mike explicitly asked for consolidation inside the repo folder once asked
  to choose; gitignore achieves the same "never gets committed" guarantee without fighting
  where he'd already put the files.
- Caution for later: this only holds as long as `transcripts/` stays in `.gitignore`. If
  that line is ever removed, everything in it would be eligible for `git add`. Don't remove
  it without re-reading this entry.
- Date: 2026-07-24
