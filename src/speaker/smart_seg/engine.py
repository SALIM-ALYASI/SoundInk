import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from .rules import SegConfig, SegResult, detect_mode, safe_ends, strip_diacritics


CfgIn = Union[Path, str, Dict[str, Any]]


def _load_cfg(cfg_in: CfgIn) -> Dict[str, Any]:
    """
    يقبل:
    - Path: ملف JSON
    - str: مسار ملف JSON
    - dict: إعدادات جاهزة
    """
    if isinstance(cfg_in, dict):
        return cfg_in

    if isinstance(cfg_in, str):
        cfg_in = Path(cfg_in)

    # هنا cfg_in صار Path
    return json.loads(cfg_in.read_text(encoding="utf-8"))


def _words(s: str) -> List[str]:
    # كلمات عربية/مختلطة بسيطة
    s = s.strip()
    if not s:
        return []
    return [w for w in re.split(r"\s+", s) if w]


def _split_by_punct(text: str) -> List[str]:
    """
    تقسيم أولي على الوقفات الطبيعية بدون قص كلمات.
    نحافظ على الترقيم داخل الجملة قدر الإمكان.
    """
    text = text.strip()
    if not text:
        return []
    # نضيف فاصل بعد علامات الوقف
    parts = re.split(r"(?<=[،…\.\!\؟\?؛:])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _protect_patterns(text: str, patterns: List[str]) -> Tuple[str, List[str]]:
    """
    يجمّد المقاطع الحساسة (مثل: اشفِ فلان وعافِ جسده) حتى لا تنكسر.
    """
    frozen: List[str] = []
    out = text

    for _, pat in enumerate(patterns):
        rx = re.compile(pat)
        while True:
            m = rx.search(out)
            if not m:
                break
            frozen_text = m.group(0)
            token = f"[[[PROT_{len(frozen):03d}]]]"
            frozen.append(frozen_text)
            out = out[:m.start()] + token + out[m.end():]

    return out, frozen


def _unprotect(s: str, frozen: List[str]) -> str:
    for idx in reversed(range(len(frozen))):
        s = s.replace(f"[[[PROT_{idx:03d}]]]", frozen[idx])
    return s


def _breath_pack(parts: List[str], max_words: int, hard_max_words: int) -> List[str]:
    """
    نجمع أجزاء قصيرة داخل chunk واحد بناءً على عدد الكلمات (تنفّس).
    """
    chunks: List[str] = []
    cur: List[str] = []
    cur_words = 0

    def flush():
        nonlocal cur, cur_words
        if cur:
            chunks.append(" ".join(cur).strip())
        cur = []
        cur_words = 0

    for p in parts:
        w = len(_words(p))
        if w == 0:
            continue

        # إذا الجزء وحده كبير جدًا -> قصّه بطريقة آمنة على كلمات
        if w > hard_max_words:
            flush()
            ws = _words(p)
            start = 0
            while start < len(ws):
                end = min(start + max_words, len(ws))
                piece = " ".join(ws[start:end]).strip()
                chunks.append(piece)
                start = end
            continue

        # محاولة ضمّه للـ current chunk
        if cur_words + w <= max_words:
            cur.append(p)
            cur_words += w
        else:
            flush()
            cur.append(p)
            cur_words = w

    flush()
    return chunks


def _compute_pads(chunks: List[str], mode_cfg: SegConfig) -> List[int]:
    """
    يحدد السكوت بين المقاطع حسب نهاية الـ chunk (، / … / نهاية فقرة).
    """
    pads: List[int] = []
    for i in range(len(chunks) - 1):
        a = chunks[i].strip()

        if a.endswith("،"):
            pads.append(mode_cfg.pad_ms_short)
        elif a.endswith("…"):
            pads.append(mode_cfg.pad_ms_med)
        else:
            pads.append(mode_cfg.pad_ms_med)
    return pads


def segment_text(text: str, cfg_in: CfgIn) -> SegResult:
    cfg = _load_cfg(cfg_in)

    dua_keywords = cfg.get("dua_keywords", [])
    patterns = cfg.get("protected_patterns", [])

    mode = detect_mode(text, dua_keywords)
    mode_cfg_raw = cfg["modes"][mode]
    mode_cfg = SegConfig(**mode_cfg_raw)

    # 1) freeze sensitive phrases
    tmp, frozen = _protect_patterns(text, patterns)

    # 2) split by punctuation/breaks
    parts = []
    for block in tmp.split("\n"):
        block = block.strip()
        if not block:
            continue
        parts.extend(_split_by_punct(block))

    # 3) breath-aware packing
    chunks = _breath_pack(
        parts,
        max_words=mode_cfg.max_words,
        hard_max_words=mode_cfg.hard_max_words,
    )

    # 4) unfreeze + smooth endings
    chunks = [_unprotect(c, frozen) for c in chunks]
    chunks = [safe_ends(c) for c in chunks if c.strip()]

    pads = _compute_pads(chunks, mode_cfg)

    return SegResult(mode=mode, chunks=chunks, pad_ms=pads)


def export_debug(result: SegResult) -> str:
    return json.dumps(asdict(result), ensure_ascii=False, indent=2)