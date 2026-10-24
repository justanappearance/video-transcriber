"""Turn a <stem>.captions.json (written by transcribe.py) into a transparent caption overlay
video: the whole cue on screen, with a box behind the word currently being spoken - the
CapCut / social-media "highlighted word" look. Drop the .mov on a track above the clip in
Resolve, starting at the same point as the audio it was transcribed from.

ProRes 4444 with alpha, not an .srt or ASS file - Resolve's subtitle track can only put a
background behind a whole cue, never one word inside it. See DECISIONS.md.
"""

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FRAME_WIDTH = 1080
FRAME_HEIGHT = 1920
FPS = 30
FONT_PATH = r"C:\Windows\Fonts\seguibl.ttf"  # Segoe UI Black - heavy, ships with Windows
FONT_SIZE = 84
TEXT_COLOR = (255, 255, 255, 255)
OUTLINE_COLOR = (0, 0, 0, 255)
OUTLINE_WIDTH = 5
HIGHLIGHT_COLOR = (0, 0, 0, 255)
HIGHLIGHT_PAD_X = 14
HIGHLIGHT_PAD_Y = 10
HIGHLIGHT_RADIUS = 14
MAX_LINE_WIDTH = int(FRAME_WIDTH * 0.88)
# Lower-middle of a vertical frame: clear of the face, above the platform UI at the bottom.
CAPTION_CENTER_Y = int(FRAME_HEIGHT * 0.68)
LINE_SPACING = int(FONT_SIZE * 1.25)


def _layout_words(words, font):
    """Word-wraps a cue by pixel width and returns each word's (x, baseline_y)."""
    space = font.getlength(" ")
    lines, current, current_width = [], [], 0.0
    for word in words:
        width = font.getlength(word["text"])
        needed = width if not current else current_width + space + width
        if current and needed > MAX_LINE_WIDTH:
            lines.append((current, current_width))
            current, current_width = [], 0.0
            needed = width
        current.append((word, width))
        current_width = needed
    if current:
        lines.append((current, current_width))

    positions = []
    first_baseline = CAPTION_CENTER_Y - (len(lines) - 1) * LINE_SPACING / 2
    for line_index, (line, line_width) in enumerate(lines):
        x = (FRAME_WIDTH - line_width) / 2
        baseline = first_baseline + line_index * LINE_SPACING
        for word, width in line:
            positions.append((x, baseline, width))
            x += width + space
    return positions


def _draw_frame(cue, active_index, font, box_top, box_bottom):
    image = Image.new("RGBA", (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    positions = _layout_words(cue["words"], font)
    if active_index is not None:
        x, baseline, width = positions[active_index]
        draw.rounded_rectangle(
            (x - HIGHLIGHT_PAD_X, baseline + box_top - HIGHLIGHT_PAD_Y,
             x + width + HIGHLIGHT_PAD_X, baseline + box_bottom + HIGHLIGHT_PAD_Y),
            radius=HIGHLIGHT_RADIUS, fill=HIGHLIGHT_COLOR,
        )
    for word, (x, baseline, _) in zip(cue["words"], positions):
        draw.text((x, baseline), word["text"], font=font, fill=TEXT_COLOR, anchor="ls",
                  stroke_width=OUTLINE_WIDTH, stroke_fill=OUTLINE_COLOR)
    return image.tobytes()


def _active_word_index(cue, t):
    """The highlight sits on the most recently started word and stays there through the
    pause before the next one - jumping back to "nothing highlighted" in every small gap
    between words reads as flicker."""
    active = None
    for i, word in enumerate(cue["words"]):
        if word["start"] <= t:
            active = i
    return active


def render(captions_path: Path, out_path: Path):
    cues = json.loads(captions_path.read_text(encoding="utf-8"))
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    # Box height comes from fixed reference glyphs, not each word's own bbox, so every
    # word's box is the same height ("a" and "Hey" would otherwise get different boxes).
    box_top = font.getbbox("H", anchor="ls")[1]
    box_bottom = font.getbbox("g", anchor="ls")[3]

    total_frames = int(cues[-1]["end"] * FPS) + 1 if cues else 0
    blank = Image.new("RGBA", (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0, 0)).tobytes()
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
         "-s", f"{FRAME_WIDTH}x{FRAME_HEIGHT}", "-r", str(FPS), "-i", "-",
         "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
         "-alpha_bits", "16", "-vendor", "apl0", str(out_path)],
        stdin=subprocess.PIPE,
    )
    # Frames only change when the cue or highlighted word changes, so render each distinct
    # state once and re-send the same bytes for every frame it covers.
    rendered = {}
    cue_index = 0
    for frame in range(total_frames):
        t = frame / FPS
        while cue_index < len(cues) and cues[cue_index]["end"] <= t:
            cue_index += 1
        if cue_index >= len(cues) or cues[cue_index]["start"] > t:
            ffmpeg.stdin.write(blank)
            continue
        cue = cues[cue_index]
        state = (cue_index, _active_word_index(cue, t))
        if state not in rendered:
            rendered = {state: _draw_frame(cue, state[1], font, box_top, box_bottom)}
        ffmpeg.stdin.write(rendered[state])
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError("ffmpeg failed to encode the caption overlay")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a highlighted-word caption overlay.")
    parser.add_argument("captions", type=Path, help="<stem>.captions.json from transcribe.py")
    args = parser.parse_args()
    out = args.captions.with_name(args.captions.name.replace(".captions.json", ".captions.mov"))
    print(f"Caption overlay written to {render(args.captions, out)}")
