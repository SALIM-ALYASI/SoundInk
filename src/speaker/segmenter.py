from __future__ import annotations

import re
from typing import TypedDict


PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\؟\?\؛\:…])\s+")
MULTI_SPACE_RE = re.compile(r"\s+")


class TextSegment(TypedDict):
    text: str
    is_paragraph_break: bool


def normalize_segment_text(text: str) -> str:
    text = (text or "").strip()
    text = MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def split_text_into_segments(text: str, max_chars: int = 220) -> list[str]:
    """
    تبقى للتوافق مع بقية المشروع.
    ترجع النصوص فقط بدون metadata.
    """
    segments = split_text_into_segments_with_breaks(text, max_chars=max_chars)
    return [seg["text"] for seg in segments]


def split_text_into_segments_with_breaks(
    text: str,
    max_chars: int = 220,
) -> list[TextSegment]:
    """
    تقسيم النص إلى مقاطع صوتية مع الحفاظ على الفقرات.
    كل مقطع يرجع معه:
    - text
    - is_paragraph_break
    """

    text = (text or "").strip()
    if not text:
        return []

    paragraphs = extract_paragraphs(text)
    segments: list[TextSegment] = []

    for paragraph in paragraphs:
        paragraph_segments = split_single_paragraph(paragraph, max_chars=max_chars)

        for i, seg in enumerate(paragraph_segments):
            segments.append(
                {
                    "text": seg,
                    "is_paragraph_break": i == len(paragraph_segments) - 1,
                }
            )

    return segments


def extract_paragraphs(text: str) -> list[str]:
    paragraphs = [normalize_segment_text(p) for p in PARAGRAPH_SPLIT_RE.split(text)]
    return [p for p in paragraphs if p]


def split_single_paragraph(text: str, max_chars: int = 220) -> list[str]:
    sentence_parts = split_paragraph_into_sentences(text)

    if not sentence_parts:
        cleaned = normalize_segment_text(text)
        return [cleaned] if cleaned else []

    current = ""
    paragraph_segments: list[str] = []

    for sentence in sentence_parts:
        sentence = normalize_segment_text(sentence)
        if not sentence:
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            paragraph_segments.append(current)
            current = ""

        if len(sentence) <= max_chars:
            current = sentence
            continue

        forced_segments = force_split_long_text(sentence, max_chars=max_chars)

        if not forced_segments:
            continue

        paragraph_segments.extend(forced_segments[:-1])
        current = forced_segments[-1]

    if current:
        paragraph_segments.append(current)

    return [normalize_segment_text(seg) for seg in paragraph_segments if normalize_segment_text(seg)]


def split_paragraph_into_sentences(text: str) -> list[str]:
    """
    تقسيم الفقرة إلى جمل اعتمادًا على علامات الوقف الأساسية.
    """
    text = normalize_segment_text(text)
    if not text:
        return []

    parts = SENTENCE_SPLIT_RE.split(text)
    cleaned_parts = [normalize_segment_text(p) for p in parts if normalize_segment_text(p)]

    return cleaned_parts


def force_split_long_text(text: str, max_chars: int) -> list[str]:
    """
    إذا كانت الجملة طويلة جدًا:
    1) نحاول تقسيمها على الفواصل أولًا
    2) ثم على الكلمات إذا بقيت طويلة
    """
    text = normalize_segment_text(text)
    if not text:
        return []

    comma_chunks = split_by_soft_punctuation(text, max_chars=max_chars)
    final_chunks: list[str] = []

    for chunk in comma_chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
            continue

        final_chunks.extend(force_split_by_words(chunk, max_chars=max_chars))

    return [normalize_segment_text(c) for c in final_chunks if normalize_segment_text(c)]


def split_by_soft_punctuation(text: str, max_chars: int) -> list[str]:
    """
    محاولة تقسيم ناعم باستخدام الفاصلة العربية/الإنجليزية
    بدل القص المباشر على الكلمات.
    """
    parts = re.split(r"(?<=[،,])\s+", text)
    parts = [normalize_segment_text(p) for p in parts if normalize_segment_text(p)]

    if not parts:
        return []

    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = f"{current} {part}".strip() if current else part

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = part

    if current:
        chunks.append(current)

    return chunks


def force_split_by_words(text: str, max_chars: int) -> list[str]:
    words = text.split()

    chunks: list[str] = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip() if current else word

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(word) <= max_chars:
            current = word
        else:
            hard_parts = split_very_long_token(word, max_chars=max_chars)
            chunks.extend(hard_parts[:-1])
            current = hard_parts[-1] if hard_parts else ""

    if current:
        chunks.append(current)

    return chunks


def split_very_long_token(text: str, max_chars: int) -> list[str]:
    """
    آخر حل: تقسيم كلمة/رمز طويل جدًا لوحده.
    """
    if not text:
        return []

    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]