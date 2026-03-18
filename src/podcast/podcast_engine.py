from __future__ import annotations

import re
from typing import Dict, List

from src.podcast.emotion_engine import apply_emotion


ARABIC_DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")


def normalize_text(text: str) -> str:
    text = ARABIC_DIACRITICS_RE.sub("", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


TITLE_KEYWORDS = ["عنوان", "عنوان الحلقة"]
OUTRO_KEYWORDS = ["الخاتمة"]
INTRO_KEYWORDS = ["المقدمة"]


def detect_section_type(text: str) -> str:
    t = normalize_text(text)

    if any(k in t for k in TITLE_KEYWORDS):
        return "title"

    if any(k in t for k in INTRO_KEYWORDS):
        return "intro"

    if any(k in t for k in OUTRO_KEYWORDS):
        return "outro"

    return "normal"


def split_text_blocks(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text or "")
    return [p.strip() for p in parts if p.strip()]


PAUSE_MAP = {
    "title": 1.2,
    "intro": 0.8,
    "normal": 0.5,
    "outro": 1.0,
}


def build_podcast_segments(text: str) -> List[Dict]:
    blocks = split_text_blocks(text)
    segments: List[Dict] = []

    for block in blocks:
        section_type = detect_section_type(block)
        emotion_data = apply_emotion(block)

        base_pause = PAUSE_MAP.get(section_type, 0.5)
        emotion_pause = float(emotion_data.get("pause", 0.5))

        final_pause = max(base_pause, emotion_pause)

        segments.append(
            {
                "text": emotion_data["text"],
                "type": section_type,
                "emotion": emotion_data["emotion"],
                "speed": emotion_data["speed"],
                "pause": final_pause,
            }
        )

    return segments