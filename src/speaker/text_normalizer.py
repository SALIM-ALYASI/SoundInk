from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from src.lexicon.manager import create_suggestion, load_suggestions


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEXICON_FILE = PROJECT_ROOT / "data" / "lexicon" / "pronunciation_lexicon.json"

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

# رموز نحب نشيلها من النص قبل TTS
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

TECH_CONTEXT_WORDS = {
    "ai", "model", "models", "machine", "learning",
    "data", "api", "openai", "chatgpt", "system",
    "algorithm", "neural", "network",
}

COMPANY_CONTEXT_WORDS = {
    "company", "startup", "ceo", "founded", "team",
}


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


def normalize_ellipsis(text: str) -> str:
    text = (text or "").replace("...", "…")
    text = re.sub(r"…{2,}", "…", text)
    return text


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

        # نحافظ على pause markers كما هي
        if PAUSE_MARKER_RE.match(stripped_line):
            kept_lines.append(stripped_line)
            continue

        if ENUM_LINE_RE.match(line):
            continue

        normalized = normalize_marker_text(line)
        if normalized in {
            "بودكاست",
            "اعداد وتقديم",
            "عنوان الحلقة",
            "المقدمة",
            "الفقرة الاولى",
            "الفقرة الثانية",
            "الفقرة الثالثة",
            "الفقرة الرابعة",
            "الفقرة الخامسة",
            "الخاتمة",
        }:
            continue

        kept_lines.append(line)

    return "\n".join(kept_lines).strip()


def clean_line_for_tts(line: str) -> str:
    line = strip_invisible_chars(line)
    line = strip_tatweel(line)
    line = normalize_quotes_and_symbols(line)
    line = normalize_ellipsis(line)
    line = remove_extra_spaces(line)

    # نحافظ على pause markers كما هي
    if PAUSE_MARKER_RE.match(line):
        return line

    line = normalize_punctuation_spacing(line)

    # إزالة الرموز الزائدة المنفصلة مع الإبقاء على $
    line = re.sub(
        r"[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF،؛:,.!?؟…\"'()/\-\$]",
        " ",
        line,
    )
    line = remove_extra_spaces(line)
    line = normalize_punctuation_spacing(line)

    return line.strip()


# -------------------------
# Lexicon
# -------------------------

@lru_cache(maxsize=1)
def load_lexicon() -> dict:
    if not LEXICON_FILE.exists():
        return {}

    try:
        data = json.loads(LEXICON_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return {}


@lru_cache(maxsize=1)
def build_replacement_pairs() -> list[tuple[str, str]]:
    lexicon = load_lexicon()
    pairs: list[tuple[str, str]] = []

    for _, items in lexicon.items():
        if not isinstance(items, list):
            continue

        for item in items:
            if not isinstance(item, dict):
                continue

            original = str(item.get("original", "")).strip()
            formatted = str(item.get("formatted", "")).strip()

            if original and formatted:
                pairs.append((original, formatted))

    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    return pairs


def _safe_pattern_for_phrase(phrase: str) -> re.Pattern:
    escaped = re.escape(phrase)
    has_word_chars = bool(re.search(r"[A-Za-z0-9\u0600-\u06FF]", phrase))

    if has_word_chars:
        pattern = rf"(?<![A-Za-z0-9\u0600-\u06FF]){escaped}(?=$|[\s،؛:,.!?؟…])"
    else:
        pattern = escaped

    return re.compile(pattern, re.IGNORECASE)


@lru_cache(maxsize=512)
def _compiled_pattern(phrase: str) -> re.Pattern:
    return _safe_pattern_for_phrase(phrase)


def apply_pronunciation_lexicon(text: str) -> str:
    for original, formatted in build_replacement_pairs():
        pattern = _compiled_pattern(original)
        text = pattern.sub(formatted, text)
    return text


# -------------------------
# Adaptive Engine
# -------------------------

def get_best_suggestion(word: str) -> str | None:
    suggestions = load_suggestions()
    candidates = []

    for item in suggestions:
        if item.get("status") != "approved":
            continue

        if item.get("original", "").lower() == word.lower():
            score = float(item.get("similarity", 0.0))
            count = int(item.get("count", 1))
            final_score = score + (count * 0.05)
            candidates.append((final_score, item.get("suggested")))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def apply_adaptive_pronunciation(text: str) -> str:
    words = text.split()
    result = []

    for word in words:
        if PAUSE_MARKER_RE.match(word):
            result.append(word)
            continue

        best = get_best_suggestion(word)
        result.append(best if best else word)

    return " ".join(result)


# -------------------------
# Context Engine
# -------------------------

def detect_context(words: list[str], index: int) -> str:
    window = words[max(0, index - 2): index + 3]
    normalized = [w.lower() for w in window]

    if any(w in TECH_CONTEXT_WORDS for w in normalized):
        return "tech"

    if any(w in COMPANY_CONTEXT_WORDS for w in normalized):
        return "company"

    return "general"


def context_aware_pronunciation(word: str, context: str) -> str:
    w = word.lower()

    if w == "ai":
        return "إيه آي" if context == "tech" else "اي آي"

    if w == "api":
        return "إيه بي آي"

    return word


def apply_context_pronunciation(text: str) -> str:
    words = text.split()
    result = []

    for i, word in enumerate(words):
        if PAUSE_MARKER_RE.match(word):
            result.append(word)
            continue

        context = detect_context(words, i)
        result.append(context_aware_pronunciation(word, context))

    return " ".join(result)


# -------------------------
# Smart Phonetic
# -------------------------

def smart_phonetic_en_to_ar(word: str) -> str:
    w = word.lower()

    special_map = {
        "openai": "أوبن إيه آي",
        "chatgpt": "شات جي بي تي",
        "api": "إيه بي آي",
        "ai": "إيه آي",
    }

    if w in special_map:
        return special_map[w]

    replacements = [
        ("tion", "شن"),
        ("ph", "ف"),
        ("sh", "ش"),
        ("ch", "تش"),
        ("th", "ث"),
        ("oo", "و"),
        ("ee", "ي"),
        ("ai", "اي"),
        ("ay", "اي"),
    ]

    for en, ar in replacements:
        w = w.replace(en, ar)

    char_map = {
        "a": "ا", "b": "ب", "c": "ك", "d": "د", "e": "ي", "f": "ف",
        "g": "ج", "h": "ه", "i": "ي", "j": "ج", "k": "ك", "l": "ل",
        "m": "م", "n": "ن", "o": "و", "p": "ب", "q": "ك", "r": "ر",
        "s": "س", "t": "ت", "u": "و", "v": "ف", "w": "و", "x": "كس",
        "y": "ي", "z": "ز",
    }

    return "".join(char_map.get(ch, ch) for ch in w)


# -------------------------
# Auto Learning
# -------------------------

def extract_unknown_words(text: str) -> list[str]:
    words = UNKNOWN_WORD_RE.findall(text)
    return list(dict.fromkeys(words))


def is_in_lexicon(word: str) -> bool:
    lexicon = load_lexicon()

    for items in lexicon.values():
        if not isinstance(items, list):
            continue

        for item in items:
            if not isinstance(item, dict):
                continue

            if item.get("original", "").lower() == word.lower():
                return True

    return False


def auto_learn_lexicon(text: str) -> None:
    for word in extract_unknown_words(text):
        if is_in_lexicon(word):
            continue

        try:
            create_suggestion(
                original=word,
                suggested=smart_phonetic_en_to_ar(word),
                recognized="",
                similarity=0.5,
            )
        except Exception:
            pass


# -------------------------
# Public Helpers
# -------------------------

def prepare_paragraphs_for_tts(text: str) -> list[str]:
    text = normalize_newlines(text)
    text = remove_decorative_lines(text)
    text = remove_structural_markers(text)
    text = remove_extra_spaces(text)

    paragraphs: list[str] = []

    for block in text.split("\n\n"):
        cleaned_lines: list[str] = []

        for line in block.split("\n"):
            cleaned = clean_line_for_tts(line)
            if not cleaned:
                continue

            # إذا marker، نحافظ عليه كفقرة مستقلة
            if PAUSE_MARKER_RE.match(cleaned):
                if cleaned_lines:
                    paragraph = " ".join(cleaned_lines).strip()
                    paragraph = remove_extra_spaces(paragraph)
                    paragraph = normalize_punctuation_spacing(paragraph)

                    if paragraph:
                        paragraphs.append(paragraph)

                    cleaned_lines = []

                paragraphs.append(cleaned)
                continue

            cleaned_lines.append(cleaned)

        if cleaned_lines:
            paragraph = " ".join(cleaned_lines).strip()
            paragraph = remove_extra_spaces(paragraph)
            paragraph = normalize_punctuation_spacing(paragraph)

            if paragraph:
                paragraphs.append(paragraph)

    return paragraphs


# -------------------------
# Main Pipeline
# -------------------------

def prepare_tts_text(text: str) -> str:
    paragraphs = prepare_paragraphs_for_tts(text)

    if not paragraphs:
        return ""

    text = "\n\n".join(paragraphs)

    auto_learn_lexicon(text)

    text = apply_pronunciation_lexicon(text)
    # text = apply_adaptive_pronunciation(text)
    # text = apply_context_pronunciation(text)

    paragraphs_after_processing: list[str] = []

    for paragraph in text.split("\n\n"):
        paragraph = remove_extra_spaces(paragraph)

        if PAUSE_MARKER_RE.match(paragraph):
            paragraphs_after_processing.append(paragraph)
            continue

        paragraph = normalize_punctuation_spacing(paragraph)
        if paragraph:
            paragraphs_after_processing.append(paragraph)

    text = "\n\n".join(paragraphs_after_processing)
    text = normalize_newlines(text)

    return text.strip()