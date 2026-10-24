"""After fixing misheard words in a <stem>.captions.json, run this on it to grow
vocabulary.txt from those fixes. Each word whose "text" no longer matches what Whisper
"heard" is offered as a new vocabulary word, and only added on a yes - a one-off fix
(a word Whisper doesn't usually get wrong) shouldn't become a permanent bias.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from transcribe import VOCABULARY_PATH, load_vocabulary


def _normalized(text: str) -> str:
    # Capitalization and punctuation edits aren't vocabulary corrections.
    return re.sub(r"[^\w']", "", text).lower()


def find_corrections(captions_path: Path) -> Counter:
    """Counts (heard, corrected) pairs where Mike changed the actual word."""
    corrections = Counter()
    for cue in json.loads(captions_path.read_text(encoding="utf-8")):
        for word in cue["words"]:
            heard = _normalized(word.get("heard", word["text"]))
            corrected = _normalized(word["text"])
            if corrected and heard != corrected:
                corrections[(heard, corrected)] += 1
    return corrections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add caption corrections to vocabulary.txt.")
    parser.add_argument("captions", type=Path, nargs="+", help="edited <stem>.captions.json file(s)")
    args = parser.parse_args()

    known = {word.lower() for word in load_vocabulary()}
    corrections = Counter()
    for path in args.captions:
        corrections.update(find_corrections(path))

    added = []
    for (heard, corrected), count in corrections.most_common():
        if corrected in known:
            continue
        answer = input(f'Add "{corrected}" to vocabulary? (Whisper heard "{heard}", {count}x) [y/N] ')
        if answer.strip().lower().startswith("y"):
            added.append(corrected)
            known.add(corrected)

    if added:
        with VOCABULARY_PATH.open("a", encoding="utf-8") as f:
            f.writelines(word + "\n" for word in added)
        print(f"Added to {VOCABULARY_PATH}: {', '.join(added)}")
    else:
        print("Nothing new added.")
