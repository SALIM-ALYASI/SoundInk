from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from src.lexicon.manager import create_suggestion, load_suggestions


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEXICON_FILE = PROJECT_ROOT / "data" / "lexicon" / "pronunciation_lexicon.json"
# القاموس اللي يحرره المستخدم يدويًا عبر /api/v1/lexicon (مسار مختلف عن
# LEXICON_FILE أعلاه، اللي يخص نظام "التعلّم" الآلي من مطابقة ASR).
MANUAL_LEXICON_FILE = PROJECT_ROOT / "data" / "pronunciation_lexicon.json"

# -------------------------
# Regex
# -------------------------

ARABIC_DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")
MULTI_SPACE_RE = re.compile(r"[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([،؛:,.!?؟…])")
SPACE_AFTER_PUNCT_RE = re.compile(r"([،؛:,.!?؟…])([^\s\n])")
UNKNOWN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-']+")

ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2066-\u2069]")
TATWEEL_RE = re.compile(r"ـ+")
DECORATIVE_LINE_RE = re.compile(r"^\s*[-_=*#~•·]+\s*$")
ENUM_LINE_RE = re.compile(
    r"^\s*(المقدمة|الخاتمة|عنوان الحلقة|إعداد وتقديم|الفقرة\s+(الأولى|الثانية|الثالثة|الرابعة|الخامسة))\s*:?\s*$"
)
PAUSE_MARKER_RE = re.compile(r"^\${1,2}$")

SYMBOL_TRANSLATION_TABLE = str.maketrans({
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "«": '"',
    "»": '"',
    "–": "-",
    "—": "-",
    "_": " ",
    "|": " ",
    "/": " / ",
    "\\": " ",
})

# -------------------------
# Basic Cleaning
# -------------------------

def strip_diacritics(text: str) -> str:
    return ARABIC_DIACRITICS_RE.sub("", text or "")


def normalize_newlines(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    return MULTI_NEWLINE_RE.sub("\n\n", text).strip()


def remove_extra_spaces(text: str) -> str:
    lines = []
    for line in normalize_newlines(text).split("\n"):
        cleaned = MULTI_SPACE_RE.sub(" ", line).strip()
        lines.append(cleaned)
    return "\n".join(lines).strip()


def strip_invisible_chars(text: str) -> str:
    return ZERO_WIDTH_RE.sub("", text or "")


def strip_tatweel(text: str) -> str:
    return TATWEEL_RE.sub("", text or "")


def normalize_quotes_and_symbols(text: str) -> str:
    return (text or "").translate(SYMBOL_TRANSLATION_TABLE)


# 🔥 التعديل هنا
def normalize_ellipsis(text: str) -> str:
    text = text or ""

    # تحويل أي نقاط متكررة إلى فاصلة خفيفة
    text = re.sub(r"\.{2,}", "، ", text)

    # تحويل الرمز … أيضًا
    text = re.sub(r"…{1,}", "، ", text)

    # تنظيف المسافات
    text = MULTI_SPACE_RE.sub(" ", text)

    return text.strip()


def normalize_punctuation_spacing(text: str) -> str:
    text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = SPACE_AFTER_PUNCT_RE.sub(r"\1 \2", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n\s+", "\n", text)
    return text.strip()


def remove_decorative_lines(text: str) -> str:
    kept_lines: list[str] = []

    for line in normalize_newlines(text).split("\n"):
        if DECORATIVE_LINE_RE.match(line):
            continue
        kept_lines.append(line)

    return "\n".join(kept_lines).strip()


def normalize_marker_text(text: str) -> str:
    text = strip_diacritics(text)
    text = remove_extra_spaces(text)
    return text.strip(" :.-").strip()


def remove_structural_markers(text: str) -> str:
    kept_lines: list[str] = []

    for line in normalize_newlines(text).split("\n"):
        stripped_line = (line or "").strip()

        if PAUSE_MARKER_RE.match(stripped_line):
            kept_lines.append(stripped_line)
            continue

        if ENUM_LINE_RE.match(line):
            continue

        kept_lines.append(line)

    return "\n".join(kept_lines).strip()


def clean_line_for_tts(line: str) -> str:
    line = strip_invisible_chars(line)
    line = strip_tatweel(line)
    line = normalize_quotes_and_symbols(line)
    line = normalize_ellipsis(line)
    line = remove_extra_spaces(line)

    if PAUSE_MARKER_RE.match(line):
        return line

    line = normalize_punctuation_spacing(line)

    line = re.sub(
        r"[^\w\s\u0600-\u06FF،؛:,.!?؟\"'()/\-\$]",
        " ",
        line,
    )

    line = remove_extra_spaces(line)
    line = normalize_punctuation_spacing(line)

    return line.strip()


# -------------------------
# Lexicon substitution
# -------------------------

def _load_lexicon_entries() -> list[tuple[str, str]]:
    """يجمع كل مدخلات النطق (اليدوية من data/pronunciation_lexicon.json
    + المتعلَّمة آليًا من data/lexicon/pronunciation_lexicon.json) في قائمة
    واحدة (original, formatted)، بدون أي منهما كان يُستخدم فعليًا قبل هذا."""
    entries: list[tuple[str, str]] = []

    for path in (MANUAL_LEXICON_FILE, LEXICON_FILE):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        for category_items in data.values():
            if not isinstance(category_items, list):
                continue
            for item in category_items:
                if not isinstance(item, dict):
                    continue
                original = str(item.get("original") or "").strip()
                formatted = str(item.get("formatted") or "").strip()
                if original and formatted and original != formatted:
                    entries.append((original, formatted))

    # الأطول أولًا، حتى لا يأكل مدخل قصير ("قبعة") جزءًا من مدخل أطول
    # يحتويه ("القبعة") قبل أن تتاح الفرصة للمدخل الأطول نفسه.
    entries.sort(key=lambda pair: len(pair[0]), reverse=True)
    return entries


def apply_lexicon(text: str) -> str:
    if not text:
        return text

    for original, formatted in _load_lexicon_entries():
        if original in text:
            text = text.replace(original, formatted)

    return text


# -------------------------
# Public API
# -------------------------

def prepare_tts_text(text: str) -> str:
    text = normalize_newlines(text)
    text = remove_decorative_lines(text)
    text = remove_structural_markers(text)
    text = clean_line_for_tts(text)
    text = apply_lexicon(text)

    return text.strip()