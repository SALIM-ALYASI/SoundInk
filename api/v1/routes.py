from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pydub import AudioSegment

from api.v1.schemas import PodcastRequest, SegmentRequest, SpeakRequest
from core.audio_merger import merge_wav_files, mix_voice_with_bgm_ducking
from core.voice_registry import get_voice, list_voices
from src.lexicon.manager import (
    approve_suggestion,
    grouped_suggestions,
    mark_suggestion_needs_edit,
    reject_suggestion,
)
from src.speaker.inference_manager import tts_manager
from src.speaker.segmenter import split_text_into_segments_with_breaks
from src.speaker.text_normalizer import prepare_tts_text

router = APIRouter()

DATA_DIR = Path("data")
LEXICON_FILE = DATA_DIR / "lexicon" / "pronunciation_lexicon.json"
BRAIN_DIR = DATA_DIR / "brain"
BGM_DIR = Path("assets") / "bgm"

DEFAULT_LEXICON = {
    "misc_pronunciation": [],
    "names_pronunciation": [],
    "tribes_pronunciation": [],
}

ARABIC_DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")

INTRO_TEXT = """
أهلاً بكم…
هذا بودكاست رَسيس مع سالم الحَجري،
حيث نقترب من الأفكار… ونترك أثرًا يبقى.
"""

OUTRO_TEXT = """
وبين كل فكرةٍ وأخرى…
يبقى الأثر.

… هذا كان بودكاست رَسيس
إلى لقاءٍ قريب.
"""

SKIP_MARKERS = [
    "بودكاست",
    "إعداد وتقديم",
    "عنوان الحلقة",
    "المقدمة",
    "الفقرة الأولى",
    "الفقرة الثانية",
    "الفقرة الثالثة",
    "الفقرة الرابعة",
    "الفقرة الخامسة",
    "الخاتمة",
]

PAUSE_MARKER_TO_MS = {
    "$": 3000,
    "$$": 6000,
}


class SuggestionActionRequest(BaseModel):
    suggestion_id: str = Field(..., min_length=1)


class SuggestionEditRequest(BaseModel):
    suggestion_id: str = Field(..., min_length=1)
    suggested: str = Field(..., min_length=1)


# -----------------------------
# LEXICON HELPERS
# -----------------------------
def load_lexicon() -> dict[str, list[dict[str, str]]]:
    if not LEXICON_FILE.exists():
        return dict(DEFAULT_LEXICON)

    try:
        data = json.loads(LEXICON_FILE.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_LEXICON)

    if not isinstance(data, dict):
        return dict(DEFAULT_LEXICON)

    for key, default_value in DEFAULT_LEXICON.items():
        if key not in data or not isinstance(data[key], list):
            data[key] = default_value.copy()

    return data


def save_lexicon(data: dict[str, Any]) -> None:
    LEXICON_FILE.parent.mkdir(parents=True, exist_ok=True)
    LEXICON_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# -----------------------------
# GENERAL HELPERS
# -----------------------------
def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w\-]+", "_", (name or "").strip(), flags=re.UNICODE)
    return cleaned.strip("_") or "podcast_episode"


def delete_file_later(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as exc:
        print(f"Failed to delete temp file: {path} -> {exc}")


def delete_files_later(paths: list[str]) -> None:
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception as exc:
            print(f"Failed to delete temp file: {path} -> {exc}")


def normalize_text_for_tts(text: str) -> str:
    return prepare_tts_text(text or "")


def read_json_file(path: Path) -> Any:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def strip_arabic_diacritics(text: str) -> str:
    return ARABIC_DIACRITICS_RE.sub("", text or "")


def normalize_marker_text(text: str) -> str:
    text = strip_arabic_diacritics(text)
    return re.sub(r"\s+", " ", text).strip()


def is_skip_marker(text: str) -> bool:
    normalized_text = normalize_marker_text(text)
    normalized_markers = {
        normalize_marker_text(marker)
        for marker in SKIP_MARKERS
    }
    return normalized_text in normalized_markers


def is_pause_marker(text: str) -> bool:
    return (text or "").strip() in PAUSE_MARKER_TO_MS


def get_pause_duration_from_marker(text: str) -> int | None:
    return PAUSE_MARKER_TO_MS.get((text or "").strip())


def find_bgm_file(bgm_id: str) -> Path | None:
    if not bgm_id:
        return None

    for ext in (".wav", ".mp3"):
        candidate = BGM_DIR / f"{bgm_id}{ext}"
        if candidate.exists():
            return candidate

    return None


def build_podcast_text(raw_text: str) -> str:
    return (
        f"{INTRO_TEXT.strip()}\n\n"
        f"{raw_text.strip()}\n\n"
        f"{OUTRO_TEXT.strip()}"
    )


def build_preview_segments(
    clean_text: str,
    max_chars: int = 220,
) -> list[dict[str, Any]]:
    raw_segments = split_text_into_segments_with_breaks(
        clean_text,
        max_chars=max_chars,
    )

    print("RAW SEGMENTS FROM SEGMENTER:")
    for i, seg in enumerate(raw_segments, start=1):
        print(i, repr(seg))

    normalized_segments: list[dict[str, Any]] = []

    for seg in raw_segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        pause_ms = get_pause_duration_from_marker(text)

        if pause_ms is not None:
            normalized_segments.append(
                {
                    "text": text,
                    "is_paragraph_break": True,
                    "type": "pause_marker",
                    "pause_ms": pause_ms,
                }
            )
            continue

        normalized_segments.append(
            {
                "text": text,
                "is_paragraph_break": bool(seg.get("is_paragraph_break", False)),
                "type": "skip_marker" if is_skip_marker(text) else "normal",
            }
        )

    print("NORMALIZED PREVIEW SEGMENTS:")
    for i, seg in enumerate(normalized_segments, start=1):
        print(i, repr(seg))

    return normalized_segments


def build_merge_inputs_from_segments(
    segments: list[dict[str, Any]],
    voice_id: str,
    skip_marker_pause_ms: int,
) -> tuple[list[Any], list[str]]:
    merge_inputs: list[Any] = []
    temp_paths: list[str] = []

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        seg_type = seg.get("type")

        if seg_type == "pause_marker" and is_pause_marker(text):
            merge_inputs.append(text)
            continue

        if seg_type == "skip_marker":
            merge_inputs.append(
                {
                    "path": None,
                    "is_paragraph_break": True,
                    "silence_ms": int(skip_marker_pause_ms),
                }
            )
            continue

        audio_path = tts_manager.generate_temp_audio(
            text=text,
            voice_id=voice_id,
        )

        if audio_path in PAUSE_MARKER_TO_MS:
            merge_inputs.append(audio_path)
            continue

        temp_paths.append(audio_path)
        merge_inputs.append(audio_path)

    print("MERGE INPUTS:")
    for i, item in enumerate(merge_inputs, start=1):
        print(i, repr(item))

    return merge_inputs, temp_paths


# -----------------------------
# LEXICON PREVIEW
# -----------------------------
@router.get("/lexicon/preview")
async def preview_pronunciation(
    text: str,
    background_tasks: BackgroundTasks,
    voice: str = "voice1",
):
    preview_text = normalize_text_for_tts(text)

    if not preview_text:
        raise HTTPException(status_code=400, detail="Text is required")

    if is_pause_marker(preview_text):
        raise HTTPException(
            status_code=400,
            detail="Pause markers cannot be previewed as standalone speech.",
        )

    try:
        audio_path = tts_manager.generate_temp_audio(
            text=preview_text,
            voice_id=voice,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Preview generation failed: {exc}",
        ) from exc

    background_tasks.add_task(delete_file_later, audio_path)

    return FileResponse(
        path=audio_path,
        media_type="audio/wav",
        filename="preview.wav",
    )


# -----------------------------
# LEXICON API
# -----------------------------
@router.get("/lexicon")
async def get_lexicon():
    return {
        "status": "ok",
        "lexicon": load_lexicon(),
    }


@router.post("/lexicon/add")
async def add_lexicon_word(
    original: str,
    formatted: str,
    category: str = "misc_pronunciation",
):
    original = (original or "").strip()
    formatted = (formatted or "").strip()
    category = (category or "").strip()

    if not original or not formatted:
        raise HTTPException(
            status_code=400,
            detail="Original and formatted are required",
        )

    data = load_lexicon()

    if category not in data:
        data[category] = []

    for item in data[category]:
        if item.get("original") == original:
            raise HTTPException(
                status_code=409,
                detail="Word already exists in this category",
            )

    entry = {
        "original": original,
        "formatted": formatted,
    }

    data[category].append(entry)
    save_lexicon(data)

    return {
        "status": "ok",
        "entry": entry,
    }


@router.post("/lexicon/update")
async def update_lexicon_word(
    original: str,
    formatted: str,
    category: str = "misc_pronunciation",
):
    original = (original or "").strip()
    formatted = (formatted or "").strip()
    category = (category or "").strip()

    if not original or not formatted:
        raise HTTPException(
            status_code=400,
            detail="Original and formatted are required",
        )

    data = load_lexicon()

    if category not in data:
        raise HTTPException(status_code=404, detail="Category not found")

    updated = None

    for item in data[category]:
        if item.get("original") == original:
            item["formatted"] = formatted
            updated = item
            break

    if not updated:
        raise HTTPException(status_code=404, detail="Word not found")

    save_lexicon(data)

    return {
        "status": "ok",
        "entry": updated,
    }


@router.post("/lexicon/delete")
async def delete_lexicon_word(
    original: str,
    category: str = "misc_pronunciation",
):
    original = (original or "").strip()
    category = (category or "").strip()

    if not original:
        raise HTTPException(status_code=400, detail="Original is required")

    data = load_lexicon()

    if category not in data:
        raise HTTPException(status_code=404, detail="Category not found")

    before_count = len(data[category])
    data[category] = [
        item for item in data[category]
        if item.get("original") != original
    ]

    if len(data[category]) == before_count:
        raise HTTPException(status_code=404, detail="Word not found")

    save_lexicon(data)

    return {"status": "ok"}


# -----------------------------
# SUGGESTIONS API
# -----------------------------
@router.get("/lexicon/suggestions")
async def get_lexicon_suggestions():
    data = grouped_suggestions()

    pending_count = len(data.get("pending", []))
    needs_edit_count = len(data.get("needs_edit", []))
    has_attention_items = (pending_count + needs_edit_count) > 0

    return {
        "status": "ok",
        "suggestions": data,
        "counts": {
            "pending": pending_count,
            "approved": len(data.get("approved", [])),
            "rejected": len(data.get("rejected", [])),
            "needs_edit": needs_edit_count,
        },
        "has_attention_items": has_attention_items,
    }


@router.post("/lexicon/suggestions/approve")
async def approve_lexicon_suggestion(payload: SuggestionActionRequest):
    item = approve_suggestion(payload.suggestion_id)

    if not item:
        raise HTTPException(status_code=404, detail="Suggestion not found")

    return {
        "status": "ok",
        "item": item,
    }


@router.post("/lexicon/suggestions/reject")
async def reject_lexicon_suggestion(payload: SuggestionActionRequest):
    item = reject_suggestion(payload.suggestion_id)

    if not item:
        raise HTTPException(status_code=404, detail="Suggestion not found")

    return {
        "status": "ok",
        "item": item,
    }


@router.post("/lexicon/suggestions/edit")
async def edit_lexicon_suggestion(payload: SuggestionEditRequest):
    item = mark_suggestion_needs_edit(
        suggestion_id=payload.suggestion_id,
        suggested=payload.suggested.strip(),
    )

    if not item:
        raise HTTPException(status_code=404, detail="Suggestion not found")

    return {
        "status": "ok",
        "item": item,
    }


# -----------------------------
# HEALTH
# -----------------------------
@router.get("/health")
async def health():
    return {
        "status": "ok",
        "project": "APVA Next",
    }


# -----------------------------
# VOICES
# -----------------------------
@router.get("/voices")
async def voices():
    return {
        "default_voice": get_voice(),
        "voices": list_voices(),
    }


@router.get("/brain/summary")
async def brain_summary():
    return {
        "graph": read_json_file(BRAIN_DIR / "project_graph.json"),
        "issues": read_json_file(BRAIN_DIR / "issues_detected.json"),
        "cleanup": read_json_file(BRAIN_DIR / "cleanup_advisor_report.json"),
        "dependency_map": read_json_file(BRAIN_DIR / "dependency_map.json"),
        "architecture": read_json_file(BRAIN_DIR / "architecture_report.json"),
        "restructure": read_json_file(BRAIN_DIR / "restructure_plan.json"),
    }


# -----------------------------
# BASIC SPEAK
# -----------------------------
@router.post("/speak")
async def speak(
    payload: SpeakRequest,
    background_tasks: BackgroundTasks,
):
    raw_text = (payload.text or "").strip()

    if not raw_text:
        raise HTTPException(status_code=400, detail="Text is required")

    preview_text = normalize_text_for_tts(raw_text[:220])

    if not preview_text:
        raise HTTPException(status_code=400, detail="Prepared text is empty")

    if is_pause_marker(preview_text):
        raise HTTPException(
            status_code=400,
            detail="Pause marker alone cannot be converted to speech.",
        )

    try:
        audio_path = tts_manager.generate_temp_audio(
            text=preview_text,
            voice_id=payload.voice_id,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Speech generation failed: {exc}",
        ) from exc

    background_tasks.add_task(delete_file_later, audio_path)

    return FileResponse(
        path=audio_path,
        media_type="audio/wav",
        filename="speech.wav",
    )


# -----------------------------
# SEGMENTS PREVIEW
# -----------------------------
@router.post("/segments/preview")
async def segments_preview(payload: SegmentRequest):
    raw_text = (payload.text or "").strip()

    if not raw_text:
        raise HTTPException(status_code=400, detail="Text is required")

    clean_text = normalize_text_for_tts(raw_text)

    print("SEGMENTS PREVIEW - CLEAN TEXT:")
    print(clean_text)

    segments = build_preview_segments(clean_text, max_chars=220)

    return {
        "status": "ok",
        "count": len(segments),
        "segments": segments,
    }


# -----------------------------
# PODCAST ENGINE
# Stable route:
# request -> clean -> segment -> generate -> merge -> bgm
# -----------------------------
@router.post("/podcast")
async def podcast(
    payload: PodcastRequest,
    background_tasks: BackgroundTasks,
):
    raw_text = (payload.text or "").strip()

    if not raw_text:
        raise HTTPException(status_code=400, detail="Text is required")

    full_text = build_podcast_text(raw_text)
    clean_text = normalize_text_for_tts(full_text)

    print("PODCAST - RAW TEXT:")
    print(raw_text)
    print("PODCAST - FULL TEXT:")
    print(full_text)
    print("PODCAST - CLEAN TEXT:")
    print(clean_text)

    if not clean_text:
        raise HTTPException(status_code=400, detail="Prepared text is empty")

    preview_segments = build_preview_segments(clean_text, max_chars=220)

    if not preview_segments:
        raise HTTPException(status_code=400, detail="No segments generated")

    try:
        merge_inputs, temp_paths = build_merge_inputs_from_segments(
            segments=preview_segments,
            voice_id=payload.voice_id,
            skip_marker_pause_ms=payload.skip_marker_pause_ms,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"TTS failed: {exc}",
        ) from exc

    if not merge_inputs:
        delete_files_later(temp_paths)
        raise HTTPException(status_code=400, detail="No audio chunks generated")

    try:
        merged = merge_wav_files(
            chunks=merge_inputs,
            silence_between_segments_ms=payload.silence_between_segments_ms,
            silence_between_paragraphs_ms=payload.silence_between_paragraphs_ms,
        )
    except Exception as exc:
        delete_files_later(temp_paths)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to merge audio: {exc}",
        ) from exc

    final_path = merged
    generated_files_to_cleanup = [*temp_paths, merged]

    if payload.intro_lead_ms:
        try:
            audio = AudioSegment.from_file(final_path)
            audio = AudioSegment.silent(duration=payload.intro_lead_ms) + audio
            audio.export(final_path, format="wav")
        except Exception as exc:
            delete_files_later(generated_files_to_cleanup)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to add intro lead: {exc}",
            ) from exc

    if payload.bgm_id:
        bgm = find_bgm_file(payload.bgm_id)
        if bgm:
            try:
                mixed_path = mix_voice_with_bgm_ducking(
                    voice_path=final_path,
                    bgm_path=str(bgm),
                )
                final_path = mixed_path
                if mixed_path not in generated_files_to_cleanup:
                    generated_files_to_cleanup.append(mixed_path)
            except Exception as exc:
                delete_files_later(generated_files_to_cleanup)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to mix background music: {exc}",
                ) from exc

    background_tasks.add_task(delete_files_later, generated_files_to_cleanup)

    return FileResponse(
        path=final_path,
        media_type="audio/wav",
        filename=f"{safe_filename(payload.episode_title)}.wav",
    )