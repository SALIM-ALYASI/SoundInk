from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple
import re

from src.lexicon.manager import add_lexicon_entry, load_lexicon, save_lexicon

# ------------------------
# CONFIG
# ------------------------
LOG_FILE = Path("data/retry_feedback.jsonl")

ARABIC_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")
MULTISPACE = re.compile(r"\s+")


# ------------------------
# MODELS
# ------------------------
@dataclass
class Change:
    original: str
    formatted: str
    type: str
    confidence: float


# ------------------------
# HELPERS
# ------------------------
def clean(text: str) -> str:
    return MULTISPACE.sub(" ", (text or "").strip())


def strip_diacritics(text: str) -> str:
    return ARABIC_DIACRITICS.sub("", text or "")


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def is_valid_change(old: str, new: str, conf: float) -> bool:
    old_n = strip_diacritics(old)
    new_n = strip_diacritics(new)

    # نفس الكلمة مع تشكيل
    if old_n == new_n:
        return True

    # تقارب عالي
    if conf >= 0.75:
        return True

    return False


# ------------------------
# CORE
# ------------------------
def extract_changes(old: str, new: str) -> List[Change]:
    old_words = clean(old).split()
    new_words = clean(new).split()

    matcher = SequenceMatcher(None, old_words, new_words)
    changes: List[Change] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        old_part = " ".join(old_words[i1:i2])
        new_part = " ".join(new_words[j1:j2])

        if not old_part or not new_part:
            continue

        conf = similarity(strip_diacritics(old_part), strip_diacritics(new_part))

        change_type = "word"
        if len(old_part.split()) > 1 or len(new_part.split()) > 1:
            change_type = "phrase"

        if is_valid_change(old_part, new_part, conf):
            changes.append(
                Change(
                    original=old_part,
                    formatted=new_part,
                    type=change_type,
                    confidence=conf,
                )
            )

    return changes


def save_change(change: Change) -> bool:
    try:
        return bool(add_lexicon_entry({
            "original": change.original,
            "formatted": change.formatted,
            "type": change.type
        }))
    except:
        lex = load_lexicon()
        lex.setdefault("learned", []).append(asdict(change))
        save_lexicon(lex)
        return True


def log_feedback(data: Dict[str, Any]):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


# ------------------------
# MAIN
# ------------------------
def learn_from_text_edit(old_text: str, new_text: str) -> Dict[str, Any]:
    changes = extract_changes(old_text, new_text)

    saved = 0
    learned = []

    for c in changes:
        if save_change(c):
            saved += 1
            learned.append(asdict(c))

    log_feedback({
        "old_text": old_text,
        "new_text": new_text,
        "changes": learned,
        "saved": saved,
        "time": datetime.utcnow().isoformat()
    })

    return {
        "detected": len(changes),
        "saved": saved,
        "learned": learned
    }