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

## Switched to BatchedInferencePipeline instead of plain WhisperModel.transcribe
- Decision: `transcribe.py` wraps the existing `WhisperModel` in `faster-whisper`'s
  `BatchedInferencePipeline` and calls `.transcribe(..., batch_size=16)` instead of calling
  `WhisperModel.transcribe(...)` directly.
- Why: Mike asked if the record → sync → transcribe loop could be faster overall. The
  Syncthing upload leg isn't something this tool can affect, but transcription is. Plain
  `WhisperModel.transcribe` processes VAD-detected speech chunks one at a time on the GPU;
  `BatchedInferencePipeline` runs them through the GPU in batches instead, which is a real
  throughput win with no accuracy tradeoff (it changes scheduling, not the model or
  quantization). Verified by re-transcribing an already-transcribed video and comparing
  output: word count came out within ~3% (1852 vs. 1904 words), consistent with normal
  VAD-boundary variance rather than a quality regression.
- Rejected: dropping to the `small` model for speed instead.
- Why rejected: that trades away accuracy, which matters more here than for footage-triage's
  quick-classification use case (see the first entry in this file) - batching gets a speed
  win without that tradeoff, so it's strictly better for this use case.
- Date: 2026-07-30

## Video journals now auto-transcribe via watch-inbox.ps1, reversing the earlier manual-only decision
- Decision: `C:\Videos\scripts\automatic\watch-inbox.ps1` now calls `transcribe.py`
  automatically (via `Start-Process python transcribe.py <dest> -WindowStyle Hidden`,
  output logged to `C:\Videos\scripts\logs\transcribe-auto.log`) the moment it detects and
  moves a "video journal" (10+ min camera video) into `backlog\phone\`. This lives in the
  watcher script, not this repo, but it's this repo's script being invoked, so it's logged
  here too.
- Why: on 2026-07-26 Mike explicitly said he only wanted specific files transcribed on
  request, not every journal automatically. On 2026-07-31 he reversed that once the pattern
  became clear in practice: "i find myself basically automatically using the video
  transcriber every time i take a video journal" - the manual-ask step had become pure
  overhead, not a meaningful gate. This is a reversal of a real prior decision, not a new
  one from scratch - see the entry two above (BatchedInferencePipeline) and watch-inbox.ps1
  git history around 2026-07-26 for the original manual-only reasoning.
- Rejected: keeping it manual (2026-07-26 status quo); a signal-file / "watch for this one"
  mechanism for the halfway case of "auto-transcribe just this next one."
- Why rejected: former was the exact overhead Mike was trying to eliminate; the halfway
  mechanism is more moving parts than needed now that "always" is the actual usage pattern.
- Caution for later: if this ever misfires (a long non-journal camera video), the fix is to
  tighten `$journalMinDurationSecs` or add real speech-detection in watch-inbox.ps1, not to
  revert this decision - re-check with Mike before reverting to manual-only, since that was
  explicitly superseded by his own stated usage pattern, not by a bug.
- Date: 2026-07-31

## Reduced BATCH_SIZE from 16 to 4 after a real CUDA OOM on a 31-minute video
- Decision: `BATCH_SIZE` in `transcribe.py` is now `4`, not `16`.
- Why: the 2026-07-30 batching change was verified on a ~17-minute clip only. The first
  real-world auto-transcribe run on a 31-minute video (2026-07-31, via the new watcher
  auto-trigger) crashed with `RuntimeError: CUDA failed with error out of memory` inside
  `generate_segment_batched`. `BatchedInferencePipeline` pads every segment in a batch to
  the length of the longest segment in that batch - a batch that happens to contain a
  couple of unusually long VAD-detected segments (e.g. a long uninterrupted ramble, which
  is exactly this tool's normal input) can spike memory far past what a shorter test video
  showed, even though `batch_size` itself is unchanged. This is a real capacity limit on
  this 6GB Pascal card, not a fluke - confirmed the fix by re-running the same 31-minute
  video at `batch_size=4` end-to-end successfully (3154 words, clean ending, no truncation).
- Rejected: keeping `batch_size=16`; reverting to plain `WhisperModel.transcribe` (no
  batching at all).
- Why rejected: former just OOMs again on the next long enough ramble; latter throws away
  the real speedup batching still gives at a smaller batch size - `4` still batches, just
  fewer segments at once, so there's no accuracy cost either way (same tradeoff analysis as
  the original batching entry above).
- Caution for later: this is a padding/memory-spike issue, not a hard formula - if OOM
  happens again even at `batch_size=4` (e.g. on an even longer or more continuous
  recording), drop further before assuming something else broke.
- Date: 2026-07-31

## Also save a timestamped JSON alongside the plain-text transcript
- Decision: `transcribe()` now writes `<stem>.json` (list of `{start, end, text}` per
  segment) next to the existing `<stem>.txt`, unconditionally.
- Why: the `video-journal-auto-cut` project needs to align individual transcript segments
  back to specific timestamps in the raw footage (to test whether an LLM reading the
  transcript can predict which segments Mike actually kept when editing) - the plain-text
  output alone has no timing information at all.
- Rejected: a separate transcription script inside `video-journal-auto-cut` that duplicates
  the whisper-loading/GPU setup here.
- Why rejected: this tool already owns "turn a video into a transcript" - adding timestamps
  to its output is a generically useful, backward-compatible extension (the .txt output is
  byte-for-byte unchanged), not a new concern that belongs in a different project.
- Date: 2026-08-08

## Output files are stamped with the video's own recording time, not whenever transcribe.py ran
- Decision: after writing `<stem>.txt`/`<stem>.json`, `transcribe()` sets both files' mtime
  to the source video's embedded `creation_time` (read via `ffprobe format_tags=creation_time`),
  falling back to the video file's own mtime if that tag is missing.
- Why: Mike noticed the transcripts folder sorted by Date Modified no longer matched
  recording order. Root cause: some journals now come in under the 10-min auto-transcribe
  threshold (see the two entries above - that threshold was tuned for a duration heuristic,
  not "is this actually a journal"), so they sit untranscribed until he asks by hand, often
  days later - while other journals auto-transcribe the instant they land. The transcript's
  mtime reflected *when transcribe.py happened to run*, not when the video was recorded, so
  Date-Modified sort scrambled relative to actual recording order once both paths were live
  at once. Stamping from the video's own embedded timestamp fixes this regardless of which
  path produced the transcript or how late it ran.
- Also backfilled all 58 pre-existing transcripts: 43 by reading `creation_time` off their
  still-present source video (found by recursive stem search under `C:\Videos\footage`), 15
  more whose source video no longer existed anywhere under `C:\Videos` (already
  triaged/deleted) by parsing the capture timestamp directly out of the Pixel filename
  convention (`PXL_YYYYMMDD_HHMMSSmmm`, UTC) instead.
- Rejected: telling Mike to just sort the folder by Name instead of Date Modified (filenames
  already sort chronologically since they embed the same timestamp).
- Why rejected: works for him browsing in Explorer, but doesn't fix directory-listing order
  for any script that reads the folder by mtime, and requires remembering to change the sort
  column - a file-level fix is permanent and needs no ongoing behavior change from Mike.
- Date: 2026-08-18

## Removed the per-segment JSON output entirely - video-journal-auto-cut was scrapped
- Decision: `transcribe()` no longer writes `<stem>.json` at all - only `<stem>.txt`. Deleted
  all 38 `.json` files that existed in `transcripts/`.
- Why: the JSON's only reason to exist (see the 2026-08-08 entry above) was
  `C:\Coding\video-journal-auto-cut`, which read it directly via a hardcoded `TRANSCRIPTS_DIR`
  in several of its scripts. Mike scrapped that project (2026-08-18), so the JSON had no
  remaining consumer - it was dead output, not backward-compat surface worth preserving.
  Came up while fixing the transcripts-folder ordering issue (previous entry): Mike asked
  whether the `.json` files could be moved out of the way, which would have meant updating
  6 references in a project that turned out to no longer be in use at all.
- Rejected: moving the JSON to a subfolder instead of deleting the feature outright (the
  original ask).
- Why rejected: once video-journal-auto-cut is confirmed dead, there's no reason to keep
  generating output nothing reads - reorganizing dead output is more work than removing it.
- Caution for later: if segment-level timestamps are ever needed again (e.g. a new project
  wants to align transcript text back to footage), this is a ~10-line re-add - see git
  history around 2026-08-08 for the original implementation, don't design it from scratch.
- Date: 2026-08-18

## Added `.srt` output alongside `.txt` - segment timestamps needed again, for captions
- Decision: `transcribe()` now also writes `<stem>.srt` (standard numbered-cue format with
  `HH:MM:SS,mmm --> HH:MM:SS,mmm` timestamps) next to the existing `<stem>.txt`, from the
  same pass over `segments` - no second transcription run, no new dependency (plain string
  formatting of `segment.start`/`segment.end`, already on every segment object).
- Why: Mike, 2026-10-12: wants on-screen captions burned into his "quitting the internet"
  daily videos, DaVinci Resolve-style (File > Import > Subtitle needs a real `.srt` with
  timestamps, not flat text). This is the "segment-level timestamps needed again" case the
  2026-08-18 entry above flagged - a different consumer (Resolve's subtitle import) than the
  one that was removed (video-journal-auto-cut), so re-added as `.srt` specifically rather
  than restoring the old per-segment `.json` shape, which had no consumer that wanted JSON.
- Verified: re-ran on `Oct 10 at 9-11 PM.wav` - both files appeared in `transcripts/` with
  matching text; `.srt` timestamps looked sane against the file's actual 36.9s duration.
- Known limitation: Whisper's segments can be long (one came out as a single 28-second cue
  on this test file) - fine for Resolve's basic subtitle import, but not short caption-sized
  chunks. Not addressed here since it wasn't asked for - flagged to Mike in case he wants
  segment-splitting later.
- Date: 2026-10-12

## .srt cues re-chunked using word-level timestamps, sized to match CapCut's auto-captions
- Decision: `transcribe()` now calls `.transcribe(..., word_timestamps=True)` and builds the
  `.srt` from `_chunk_words_into_captions()` - flattens every segment's `.words` into one
  list, groups them into chunks capped at `MAX_CAPTION_CHARS = 90` characters each (word
  boundaries only, never mid-word), then wraps each chunk's text to `MAX_CAPTION_LINE_CHARS
  = 38` per line with `textwrap.wrap`. The `.txt` output is untouched (still one line per
  Whisper segment - that file is for pasting a full transcript into a Claude chat, where
  segment length doesn't matter).
- Why: Mike, 2026-10-12, right after the plain per-segment `.srt` landed: "it has taken
  basically a 30s clip of me talking and put it in one long thing" - a single Whisper
  segment as one subtitle cue reads nothing like the short on-screen captions he's going
  for. First tried a flat 3-second cap per his "maybe 3s intervals" suggestion, but he then
  asked to match CapCut specifically ("let's use the same characters per caption block,
  lines, etc, and I'll adjust as needed") - CapCut has no published spec, but is consistently
  observed landing around 80-100 chars per block, 1-2 lines of ~35-40 chars each. Switched to
  character-based chunking before committing the 3s version, since nothing depended on it yet.
- Rejected: keeping the duration-based (3s) chunker. Why rejected: superseded by Mike's own
  follow-up ask before it was ever committed.
- Verified: re-ran on `Oct 10 at 9-11 PM.wav`.
- Date: 2026-10-12

## Cue duration also respects reading speed, not just how long Mike took to say it
- Decision: each cue's end time is now `max(spoken_end, start + max(MIN_CAPTION_SECONDS,
  len(text) / READING_CHARS_PER_SECOND))`, clamped to never run past the next cue's start.
  `READING_CHARS_PER_SECOND = 20` and `MIN_CAPTION_SECONDS = 5/6` are Netflix's English
  subtitle style guide numbers (adult-content reading speed and minimum cue duration) - the
  most commonly cited public reading-speed standard, used as-is rather than re-derived.
- Why: Mike, 2026-10-12, after seeing CapCut's fixed-character-count chunking: "the fact
  that display time is reading speed based if we could implement that that'd be cool" -
  without this, a short cue spoken very fast would flash by before it's readable, even
  though the text itself is within the character cap.
- Verified: synthetic test (20 words at 0.1s each, 2s spoken duration, 129 chars of text) -
  cue extended to 6.45s, matching the reading-speed floor instead of the literal 2s spoken
  duration. On the real test file (`Oct 10 at 9-11 PM.wav`) this made no visible difference,
  since Mike talks fast enough that every real cue already exceeded the reading-speed
  minimum - the floor only kicks in when needed.
- Date: 2026-10-12

## Caption chunk size cut from CapCut's 90/38 to 24/24 (one short line, not 2-3 stacked)
- Decision: `MAX_CAPTION_CHARS` and `MAX_CAPTION_LINE_CHARS` both dropped to 24 (was 90/38).
- Why: Mike, 2026-10-12, after importing the `.srt` and looking at it live in Resolve: "can
  you see how much text is on the screen it's a lot" (screenshot showed a 3-line stacked
  caption covering a third of the vertical frame). CapCut's 90-char/38-per-line numbers are
  tuned for CapCut's own smaller default caption font - at the large font size Mike's using,
  the same char counts wrap into 2-3 stacked lines that read as a wall of text, not a punchy
  single-line caption.
- Verified: re-ran on `Oct 10 at 9-11 PM.wav` - 18 single-line cues instead of 5 stacked
  3-line blocks, ~4-5 words each.
- Date: 2026-10-12

## Caption cues also break at phrase starts (capitalized words), not just the char cap
- Decision: `_chunk_words_into_captions` ends the current cue before any word
  `_is_phrase_start` flags (first letter uppercase, excluding "I"/"I'm"/"I've"/"I'll"/"I'd"),
  in addition to the existing `MAX_CAPTION_CHARS` backstop.
- Why: Mike, 2026-10-12: Whisper drops most sentence punctuation on his recordings but still
  capitalizes each new phrase; a pure char cap spliced the tail of one sentence onto the start
  of the next in one cue (e.g. "flag So y'all I've").
- Rejected: forcing a line break inside the cue at the phrase start instead of a new cue.
- Why rejected: Mike's fine with one line per cue; a new cue fixes the root cause (cue
  boundaries ignoring phrases) rather than reflowing text inside a badly-cut cue.
- Known tradeoff: proper nouns also capitalize, so names split cues too ("God", "Cooper"
  on the test file). No punctuation means no reliable way to tell them apart.
- Verified: re-ran on `Oct 10 at 9-11 PM.wav` - phrase boundaries now separate cues.
- Date: 2026-10-12

## Merge Whisper's split word tokens before chunking captions
- Decision: `_merge_split_words` joins any word token that doesn't start with a space onto the
  previous token (text concatenated, start of the first, end of the last) before
  `_chunk_words_into_captions` runs.
- Why: Mike, 2026-10-12: "self-control" came out as " self" + "-control", and the chunker
  split them across two cues. Real word tokens always start with a space, so a token without
  one is a continuation.
- Verified: re-ran on `Oct 10 at 9-11 PM.wav` - "self-control type shit" is one cue; all
  other cues unchanged.
- Date: 2026-10-12

## Highlighted-word captions rendered as a transparent overlay video, not a subtitle file
- Decision: `render_captions.py` renders `<stem>.captions.json` (per-word timings, written by
  `transcribe.py` alongside the `.srt`) into a ProRes 4444 `.mov` with alpha, drawn frame by
  frame with Pillow and piped to ffmpeg. The whole cue stays on screen; a black box sits behind
  the most recently started word (held through pauses, not cleared between words, to avoid
  flicker).
- Why: Mike, 2026-10-24, wanted the social-media look where a background box jumps word to word
  as it's spoken. Resolve's subtitle track can only put a background behind a whole cue.
- Rejected: one-word-per-cue `.srt` + Resolve's subtitle background (only the current word on
  screen, not the phrase); burning captions into a copy of the video (can't move/edit in
  Resolve); ASS karaoke tags (Resolve doesn't render ASS styling, and ASS has no per-word box);
  estimating word timings by spreading each `.srt` cue's duration across its words (cue ends are
  stretched by the reading-speed floor, so the highlight would drift late).
- Verified: `day 2.wav` - `.srt` byte-identical after the cue-building refactor; overlay is
  39.7s, 1080x1920, yuva444p; composited preview frames show the box on "Hey" at 1.2s and on
  "back" at 2.4s, matching the word timings.
- Date: 2026-10-24

## Learn Mike's vocabulary from his own caption corrections (hotwords, not replace rules)
- Decision: a gitignored `vocabulary.txt` is passed to faster-whisper as `hotwords`;
  `.captions.json` words keep Whisper's original as `"heard"` next to the editable `"text"`, and
  `learn_vocabulary.py` diffs the two after Mike edits, offering each corrected word for the
  vocabulary with a y/N confirm.
- Why: Mike, 2026-10-24, wanted the transcriber to get better from his fixes over time (e.g.
  Whisper wrote "hole" where he said "hoe").
- Rejected: automatic find-and-replace rules (hole->hoe would corrupt every real "hole");
  adding every correction without asking (one-off fixes would pile up as permanent bias);
  fine-tuning Whisper (far too heavy for the volume of corrections); storing an untouched copy
  of the original captions file to diff against ("heard" per word is the same info, one file).
- Known limit, verified on `day 2.wav`: with "hoe" in the vocabulary Whisper STILL wrote "this
  hole" - hotwords only nudge it, and "this hole" is plausible English on ambiguous gym audio.
  Expect hotwords to help most with words Whisper doesn't know (slang, names), not with
  swapping one common word for another. The rest of the transcript only shifted punctuation
  and one cue boundary. Manual edits remain the fix for cases like this.
- Date: 2026-10-24

## Stopped writing the .srt - the caption overlay replaced it
- Decision: `transcribe.py` no longer writes `<stem>.srt`; outputs are `.txt` and
  `.captions.json` only. `_srt_timestamp`, `MAX_CAPTION_LINE_CHARS` and the `textwrap` wrap
  went with it (the overlay wraps by pixel width in `render_captions.py`).
- Why: Mike, 2026-10-25: "i don't think i will need an srt file moving forward". He now uses
  the `render_captions.py` overlay instead of Resolve's subtitle import, and nothing else
  (video-transcriber or C:\Videos\scripts) reads `.srt` files.
- Rejected: keeping it behind a flag - no current consumer; git history has the writer if
  Resolve subtitle import is ever wanted again.
- Verified: re-ran `day 3 take 1.wav` to a scratch dir - only `.txt` + `.captions.json`
  written, `.captions.json` byte-identical to the previous run's. Existing `.srt` files in
  `transcripts/` left in place.
- Date: 2026-10-25

## Vocabulary hint removed - hotwords restyled whole transcripts and fixed nothing
- Decision: reversed "Learn Mike's vocabulary from his own caption corrections" (above).
  `transcribe.py` passes no `hotwords`/`initial_prompt`; `learn_vocabulary.py`,
  `vocabulary.txt` and the `"heard"` field in `.captions.json` are gone. Misheard words are
  fixed by hand in `.captions.json` (`"text"`).
- Why: measured 2026-10-25 on an 18-min journal (`PXL_20261008_022541211.TS.mp4`, no "hoe"
  in it). A bare lowercase "hoe" hint cut sentence-ending punctuation 80 -> 27 and lowercased
  "I" 36 times (Whisper copies the prompt's style; the caption chunker splits on capitals,
  so captions got worse too). A punctuated hint ("Words I use: hoe.") restored the style but
  still changed ~20 unrelated words ("hey y'all" -> "heyo", "saying" -> "thinking"). And the
  one word it was added for never flipped on `day 2.wav`. Cost on every transcript, no
  observed benefit. With no hint the output matched the watcher's original transcript
  byte-for-byte. Mike agreed: "yeah i guess switch the hint off then".
- Rejected: keeping a punctuated hint (still rewrites unrelated words); find-and-replace
  rules instead (rejected earlier - corrupts real uses of the word).
- Date: 2026-10-25

## Top-of-frame caption overlay
- Decision: `render_captions.py --position top` centers the caption at 18% of frame height and writes
  `<stem>.captions.top.mov`. The default stays `bottom` (78%, `<stem>.captions.mov`).
- Why: Mike, 2026-10-30: wants secondary captions at the top of a vertical clip. A separate filename
  lets both overlays of a clip exist side by side.
- Rejected: moving the overlay up in Resolve.
- Why rejected: works, but every clip would need the same manual Position tweak.
- Date: 2026-10-30

## captions.json is opt-in (`--captions`)
- Decision: `transcribe.py` writes only the `.txt` by default. `--captions` adds `<stem>.captions.json`
  and turns on word timestamps, which only that file uses.
- Why: Mike, 2026-11-10: asked to stop making it for every transcript. Most transcripts (every
  journal the inbox watcher auto-transcribes) never get a caption overlay.
- Rejected: a separate script to build captions.json from an existing transcript.
- Why rejected: the .txt has no word timings, so it would mean transcribing again anyway.
- Verified: on a day 6 clip, the .txt without `--captions` matched the earlier transcript
  byte for byte, and with `--captions` the .captions.json matched byte for byte too.
- Date: 2026-11-10
