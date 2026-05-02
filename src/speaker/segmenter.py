from __future__ import annotations

import re
from typing import TypedDict


PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
MULTI_SPACE_RE = re.compile(r"\s+")


class TextSegment(TypedDict):
    text: str
    is_paragraph_break: bool


def normalize_segment_text(text: str) -> str:
    text = (text or "").strip()
    text = MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def split_text_into_segments(text: str, max_chars: int = 140) -> list[str]:
    segments = split_text_into_segments_with_breaks(text, max_chars=max_chars)
    return [seg["text"] for seg in segments]


def split_text_into_segments_with_breaks(
    text: str,
    max_chars: int = 140,
) -> list[TextSegment]:
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


def split_single_paragraph(text: str, max_chars: int = 140) -> list[str]:
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

        if len(sentence) > max_chars:
            if current:
                paragraph_segments.append(current)
                current = ""

            forced_segments = force_split_long_text(sentence, max_chars=max_chars)
            paragraph_segments.extend(forced_segments)
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                paragraph_segments.append(current)
            current = sentence

    if current:
        paragraph_segments.append(current)

    return [
        normalize_segment_text(seg)
        for seg in paragraph_segments
        if normalize_segment_text(seg)
    ]


def split_paragraph_into_sentences(text: str) -> list[str]:
    """
    تقسيم الجمل مع دعم:
    . ! ؟ … :
    """
    text = normalize_segment_text(text)
    if not text:
        return []

    parts = re.split(r"(?<=[\.\!\؟\?…:])\s+", text)
    cleaned_parts = [
        normalize_segment_text(p)
        for p in parts
        if normalize_segment_text(p)
    ]

    return cleaned_parts


def force_split_long_text(text: str, max_chars: int) -> list[str]:
    text = normalize_segment_text(text)
    if not text:
        return []

    comma_chunks = split_by_soft_punctuation(text, max_chars=max_chars)
    final_chunks: list[str] = []

    for chunk in comma_chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            final_chunks.extend(force_split_by_words(chunk, max_chars=max_chars))

    return [
        normalize_segment_text(c)
        for c in final_chunks
        if normalize_segment_text(c)
    ]


def split_by_soft_punctuation(text: str, max_chars: int) -> list[str]:
    parts = re.split(r"(?<=[،,؛:])\s+", text)
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
        else:
            if current:
                chunks.append(current)
            current = word

    if current:
        chunks.append(current)

    return chunks


def split_very_long_token(text: str, max_chars: int) -> list[str]:
    if not text:
        return []

    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]