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
  `BatchedInferencePipeline`, and calls `.transcribe(..., batch_size=4, word_timestamps=True)`,
  writing one segment's text per line to `<output_dir>/<video stem>.txt`, and a separate
  `<output_dir>/<video stem>.srt` built from word-level timestamps, not Whisper's own
  segments - `_chunk_words_into_captions()` groups words into ~`MAX_CAPTION_CHARS` (90)
  chunks, wrapped to `MAX_CAPTION_LINE_CHARS` (38) per line, matching CapCut's observed
  auto-caption sizing (for importing as captions in DaVinci Resolve - File > Import >
  Subtitle). Each cue's display duration is then raised to at least a reading-speed floor
  (`READING_CHARS_PER_SECOND` = 20, `MIN_CAPTION_SECONDS` = 5/6 - Netflix's style-guide
  numbers) if the literal spoken duration would be too short to read, clamped to never
  overlap the next cue. Both files' mtimes are stamped to the source video's own recording
  time, not
  whenever the script happened to run - see DECISIONS.md ("Output files are stamped with
  the video's own recording time"). `output_dir` defaults to
  `C:\Coding\video-transcriber\transcripts` but can be overridden with `--output-dir`.
  (Used to also write a per-segment `.json` for `video-journal-auto-cut` - removed 2026-08-18
  when that project was scrapped; `.srt` added back 2026-10-12 for a different consumer
  (Resolve's subtitle import) - see DECISIONS.md.)

- `render_captions.py` - turns `<stem>.captions.json` (also written by `transcribe.py`: the
  same cues as the `.srt`, plus each word's own start/end) into `<stem>.captions.mov`, a
  transparent ProRes 4444 overlay (1080x1920, 30fps, captions centered at 78% of frame height) showing the whole cue with a black
  rounded box behind the word currently being spoken - the CapCut "highlighted word" look.
  Drop it on a video track above the clip in Resolve, aligned to the start of the audio it
  was transcribed from. Style knobs (font, size, colors, position) are constants at the top.
  To fix a misheard word, edit its `"text"` in the `.captions.json` and re-render.
  `python render_captions.py "transcripts/<stem>.captions.json"`

- `vocabulary.txt` (gitignored - personal, repo is public) - one word per line (`#` comments
  ok) that Whisper tends to mishear for Mike. `transcribe.py` passes them as faster-whisper
  `hotwords`: a bias toward those words, NOT a find-and-replace. Each word in
  `.captions.json` has `"text"` (what Mike edits) and `"heard"` (Whisper's original).
- `learn_vocabulary.py` - run on an edited `.captions.json`; offers each corrected word
  (normalized: case/punctuation edits don't count) as a vocabulary addition, y/N each.
  `python learn_vocabulary.py "transcripts/<stem>.captions.json"`
  Fixing a word in `.captions.json` doesn't touch the `.srt`/`.txt` - fix those separately
  (or ask Claude), then re-render the overlay.

## Usage note
As of 2026-07-31, `C:\Videos\scripts\automatic\watch-inbox.ps1` auto-calls this script the
moment it detects and moves a "video journal" (10+ min camera video) into
`C:\Videos\footage\backlog\phone\` - see that repo's README.md and the watcher script's own
comments. So a freshly-synced journal is often already transcribed by the time Mike asks.
When he still asks to "transcribe the most recent file" (e.g. for an older backlog file, or
if the auto-run failed - check `C:\Videos\scripts\logs\transcribe-auto.log`), that means the
newest file in `C:\Videos\footage\backlog\phone\` (see [[project_video_workflow]] in
Claude's memory for the full pipeline). Check that folder's contents sorted by modified time
rather than asking which file he means.

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
  ~5-6 minutes with the old `openai-whisper` CLI path). Batching (`BatchedInferencePipeline`)
  makes it faster still - see DECISIONS.md.
- **`BATCH_SIZE` is 4, not something larger.** It was originally 16, but that OOM'd on a
  real 31-minute video - `BatchedInferencePipeline` pads every segment in a batch to the
  longest one, so a batch with a couple of unusually long rambling segments (this tool's
  normal input) can spike VRAM well past what a short test clip suggested. If you're
  tempted to raise it for speed, re-verify against a long (30+ min) video first, not just
  a short one - see DECISIONS.md.
