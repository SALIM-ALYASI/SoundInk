from __future__ import annotations

import re

# -------------------------
# Emotion Keywords
# -------------------------

QUESTION_WORDS = {"هل", "لماذا", "كيف", "متى"}
EXCITED_WORDS = {"مذهل", "رائع", "قوي", "مستقبل", "ثورة"}
WARNING_WORDS = {"خطر", "تحذير", "مشكلة", "اختراق", "تهكير"}
CALM_WORDS = {"بهدوء", "ببساطة", "تدريجيًا"}


# -------------------------
# Detect Emotion
# -------------------------

def detect_emotion(text: str) -> str:
    t = text.strip()

    if "؟" in t or any(w in t for w in QUESTION_WORDS):
        return "question"

    if any(w in t for w in WARNING_WORDS):
        return "warning"

    if any(w in t for w in EXCITED_WORDS):
        return "excited"

    if any(w in t for w in CALM_WORDS):
        return "calm"

    return "neutral"


# -------------------------
# Emotion Style
# -------------------------

EMOTION_CONFIG = {
    "question": {
        "speed": 0.95,
        "pause": 0.7,
    },
    "excited": {
        "speed": 1.1,
        "pause": 0.4,
    },
    "warning": {
        "speed": 0.85,
        "pause": 0.9,
    },
    "calm": {
        "speed": 0.8,
        "pause": 1.0,
    },
    "neutral": {
        "speed": 1.0,
        "pause": 0.5,
    },
}


# -------------------------
# Apply Emotion
# -------------------------

def apply_emotion(text: str) -> dict:
    emotion = detect_emotion(text)
    config = EMOTION_CONFIG.get(emotion, EMOTION_CONFIG["neutral"])

    return {
        "text": text,
        "emotion": emotion,
        "speed": config["speed"],
        "pause": config["pause"],
    }