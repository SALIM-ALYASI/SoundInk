import re
from dataclasses import dataclass
from typing import List

DUA_DEFAULT_KEYWORDS = ["اللهم", "يا رب", "رب", "اغفر", "اشف", "اشفِ", "ارزق", "عاف", "عافِ"]

@dataclass
class SegConfig:
    max_words: int
    hard_max_words: int
    pad_ms_short: int
    pad_ms_med: int
    pad_ms_long: int
    allow_chatty: bool

@dataclass
class SegResult:
    mode: str
    chunks: List[str]
    pad_ms: List[int]  # length = len(chunks)-1 (padding between chunks)

_AR_DIAC_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")  # harakat

def strip_diacritics(s: str) -> str:
    return _AR_DIAC_RE.sub("", s)

def detect_mode(text: str, dua_keywords: List[str]) -> str:
    # اكتشاف بسيط وعملي: إذا ظهر أي كلمة دعاء -> dua
    t = strip_diacritics(text)
    for k in dua_keywords:
        if strip_diacritics(k) in t:
            return "dua"
    return "podcast"

def is_religious_line(line: str) -> bool:
    t = strip_diacritics(line)
    return any(w in t for w in ["اللهم", "يا رب", "سبحان", "استغفر", "الحمد"])

def safe_ends(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    if s.endswith(("…", "،")):
        return s
    return s + "…"