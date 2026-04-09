from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from core.voice_registry import get_voice, list_voices
from src.speaker.inference_manager import tts_manager
from src.speaker.pronunciation_learner import learn_from_text_edit
from src.speaker.text_normalizer import prepare_tts_text

router = APIRouter()

DATA_DIR = Path("data")
SEGMENTS_DIR = Path("outputs/segments")
SEGMENTS_FILE = DATA_DIR / "segments.json"
LEARNING_LOG_FILE = DATA_DIR / "retry_feedback.jsonl"
LEXICON_FILE = DATA_DIR / "pronunciation_lexicon.json"

DEFAULT_LEXICON = {
    "misc_pronunciation": [],
    "names_pronunciation": [],
    "tribes_pronunciation": [],
}

DATA_DIR.mkdir(parents=True, exist_ok=True)
SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

if not SEGMENTS_FILE.exists():
    SEGMENTS_FILE.write_text("[]", encoding="utf-8")

if not LEARNING_LOG_FILE.exists():
    LEARNING_LOG_FILE.write_text("", encoding="utf-8")

if not LEXICON_FILE.exists():
    LEXICON_FILE.write_text(
        json.dumps(DEFAULT_LEXICON, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

MULTISPACE_RE = re.compile(r"\s+")
ARABIC_DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")


# -----------------------------
# REQUEST MODELS
# -----------------------------
class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice_id: str | None = "voice1"


class SegmentCreateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    voice_id: str | None = "voice1"


class SegmentUpdateRequest(BaseModel):
    id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1, max_length=2000)
    voice_id: str | None = "voice1"


class SegmentDeleteRequest(BaseModel):
    id: str = Field(..., min_length=1)


class LearnFromEditRequest(BaseModel):
    old_text: str = Field(..., min_length=1)
    new_text: str = Field(..., min_length=1)


class LexiconLearnRequest(BaseModel):
    original: str = Field(..., min_length=1)
    edited: str = Field(..., min_length=1)
    category: str | None = "misc_pronunciation"


class LearningDecisionRequest(BaseModel):
    original: str = Field(..., min_length=1)
    formatted: str = Field(..., min_length=1)
    category: str | None = "misc_pronunciation"


class LearningRejectRequest(BaseModel):
    original: str = Field(..., min_length=1)
    formatted: str = Field(..., min_length=1)


# -----------------------------
# GENERAL HELPERS
# -----------------------------
def normalize_text(text: str) -> str:
    return MULTISPACE_RE.sub(" ", (text or "").strip())


def strip_arabic_diacritics(text: str) -> str:
    return ARABIC_DIACRITICS_RE.sub("", text or "")


def safe_remove_file(path: str | Path) -> None:
    try:
        path_str = str(path)
        if path_str and os.path.exists(path_str):
            os.remove(path_str)
    except Exception:
        pass


# -----------------------------
# SEGMENTS HELPERS
# -----------------------------
def load_segments() -> list[dict[str, Any]]:
    try:
        data = json.loads(SEGMENTS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_segments(data: list[dict[str, Any]]) -> None:
    SEGMENTS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def generate_id() -> str:
    return str(uuid.uuid4())[:8]


def generate_audio(text: str, voice_id: str | None, output_path: Path) -> None:
    clean_text = prepare_tts_text(text)

    if not clean_text:
        raise HTTPException(status_code=400, detail="النص بعد المعالجة أصبح فارغًا")

    temp = tts_manager.generate_temp_audio(
        text=clean_text,
        voice_id=voice_id,
    )

    if not temp or not os.path.exists(temp):
        raise HTTPException(status_code=500, detail="فشل إنشاء الملف الصوتي المؤقت")

    shutil.copyfile(temp, output_path)
    safe_remove_file(temp)


# -----------------------------
# LEARNING HELPERS
# -----------------------------
def append_learning_log(entry: dict[str, Any]) -> None:
    with open(LEARNING_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_learning_log(limit: int = 50) -> list[dict[str, Any]]:
    if not LEARNING_LOG_FILE.exists():
        return []

    try:
        lines = LEARNING_LOG_FILE.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    items: list[dict[str, Any]] = []

    for line in reversed(lines):
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                items.append(parsed)
        except Exception:
            continue

        if len(items) >= limit:
            break

    return items


def save_learning_log_items(items: list[dict[str, Any]]) -> None:
    with open(LEARNING_LOG_FILE, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def remove_learning_change(original: str, formatted: str) -> bool:
    items = load_learning_log(limit=10000)
    changed = False
    updated_items: list[dict[str, Any]] = []

    for entry in reversed(items):
        changes = entry.get("changes", [])
        if not isinstance(changes, list):
            updated_items.append(entry)
            continue

        new_changes = []
        for change in changes:
            if not isinstance(change, dict):
                continue

            c_original = normalize_text(str(change.get("original", "") or ""))
            c_formatted = normalize_text(str(change.get("formatted", "") or ""))

            if c_original == original and c_formatted == formatted:
                changed = True
                continue

            new_changes.append(change)

        if new_changes:
            entry["changes"] = new_changes
            updated_items.append(entry)
        elif "changes" in entry and changed:
            pass
        else:
            updated_items.append(entry)

    updated_items.reverse()
    save_learning_log_items(updated_items)
    return changed


def extract_learning_changes(learning: Any) -> list[dict[str, Any]]:
    if not isinstance(learning, dict):
        return []

    if isinstance(learning.get("changes"), list):
        return [c for c in learning["changes"] if isinstance(c, dict)]

    if isinstance(learning.get("learned"), list):
        return [c for c in learning["learned"] if isinstance(c, dict)]

    result = learning.get("result")
    if isinstance(result, dict):
        if isinstance(result.get("changes"), list):
            return [c for c in result["changes"] if isinstance(c, dict)]
        if isinstance(result.get("learned"), list):
            return [c for c in result["learned"] if isinstance(c, dict)]

    return []


def filter_learning_changes(
    changes: list[dict[str, Any]],
    min_confidence: float = 0.50,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []

    for change in changes:
        original = normalize_text(str(change.get("original", "") or ""))
        formatted = normalize_text(str(change.get("formatted", "") or ""))
        confidence_raw = change.get("confidence", 1)

        try:
            confidence = float(confidence_raw)
        except Exception:
            confidence = 1.0

        if not original or not formatted:
            continue

        if len(original) < 2:
            continue

        if original == formatted:
            continue

        if strip_arabic_diacritics(original) == strip_arabic_diacritics(formatted):
            filtered.append(
                {
                    **change,
                    "original": original,
                    "formatted": formatted,
                    "confidence": confidence,
                }
            )
            continue

        if confidence >= min_confidence:
            filtered.append(
                {
                    **change,
                    "original": original,
                    "formatted": formatted,
                    "confidence": confidence,
                }
            )

    return filtered


def summarize_learning(learning: Any, filtered_changes: list[dict[str, Any]]) -> dict[str, Any]:
    base: dict[str, Any] = learning if isinstance(learning, dict) else {}

    detected = (
        base.get("detected")
        or base.get("detected_changes")
        or len(extract_learning_changes(base))
    )

    return {
        "detected": detected,
        "saved": len(filtered_changes),
        "changes": filtered_changes,
        "raw": base,
    }


# -----------------------------
# LEXICON HELPERS
# -----------------------------
def load_lexicon() -> dict[str, list[dict[str, Any]]]:
    if not LEXICON_FILE.exists():
        return dict(DEFAULT_LEXICON)

    try:
        data = json.loads(LEXICON_FILE.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_LEXICON)

    if not isinstance(data, dict):
        return dict(DEFAULT_LEXICON)

    normalized = {
        "misc_pronunciation": [],
        "names_pronunciation": [],
        "tribes_pronunciation": [],
    }

    for key in DEFAULT_LEXICON.keys():
        raw_items = data.get(key, [])
        clean_items: list[dict[str, Any]] = []

        if isinstance(raw_items, list):
            for item in raw_items:
                if isinstance(item, dict) and item.get("original") and item.get("formatted"):
                    clean_items.append(item)
                elif isinstance(item, list):
                    for sub in item:
                        if isinstance(sub, dict) and sub.get("original") and sub.get("formatted"):
                            clean_items.append(sub)

        normalized[key] = clean_items

    return normalized


def save_lexicon(data: dict[str, list[dict[str, Any]]]) -> None:
    safe_data = {
        "misc_pronunciation": data.get("misc_pronunciation", []),
        "names_pronunciation": data.get("names_pronunciation", []),
        "tribes_pronunciation": data.get("tribes_pronunciation", []),
    }

    LEXICON_FILE.write_text(
        json.dumps(safe_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def lexicon_entry_exists(
    lexicon: dict[str, list[dict[str, Any]]],
    category: str,
    original: str,
    formatted: str | None = None,
) -> bool:
    items = lexicon.get(category, [])
    for item in items:
        if not isinstance(item, dict):
            continue

        item_original = normalize_text(str(item.get("original", "") or ""))
        item_formatted = normalize_text(str(item.get("formatted", "") or ""))

        if formatted is None:
            if item_original == original:
                return True
        else:
            if item_original == original and item_formatted == formatted:
                return True

    return False


def add_changes_to_lexicon(
    changes: list[dict[str, Any]],
    default_category: str = "misc_pronunciation",
) -> int:
    lexicon = load_lexicon()
    saved_count = 0

    category = default_category if default_category in lexicon else "misc_pronunciation"

    for change in changes:
        original = normalize_text(str(change.get("original", "") or ""))
        formatted = normalize_text(str(change.get("formatted", "") or ""))

        if not original or not formatted:
            continue

        if lexicon_entry_exists(lexicon, category, original, formatted):
            continue

        lexicon[category].append(
            {
                "original": original,
                "formatted": formatted,
            }
        )
        saved_count += 1

    if saved_count:
        save_lexicon(lexicon)

    return saved_count


# -----------------------------
# HEALTH
# -----------------------------
@router.get("/health")
async def health():
    return {"status": "ok"}


# -----------------------------
# VOICES
# -----------------------------
@router.get("/voices")
async def voices():
    return {
        "default_voice": get_voice(),
        "voices": list_voices(),
    }


# -----------------------------
# SPEAK
# -----------------------------
@router.post("/speak")
async def speak(payload: SpeakRequest, background_tasks: BackgroundTasks):
    text = normalize_text(payload.text)

    if not text:
        raise HTTPException(status_code=400, detail="text empty")

    audio_path = tts_manager.generate_temp_audio(
        text=prepare_tts_text(text),
        voice_id=payload.voice_id,
    )

    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=500, detail="speech generation failed")

    background_tasks.add_task(safe_remove_file, audio_path)

    return FileResponse(audio_path, media_type="audio/wav")


# -----------------------------
# SEGMENTS
# -----------------------------
@router.post("/segments/create")
async def create_segment(payload: SegmentCreateRequest):
    text = normalize_text(payload.text)

    if not text:
        raise HTTPException(status_code=400, detail="text empty")

    segment_id = generate_id()
    filename = f"{segment_id}.wav"
    path = SEGMENTS_DIR / filename

    generate_audio(text, payload.voice_id, path)

    segments = load_segments()

    segment = {
        "id": segment_id,
        "text": text,
        "filename": filename,
    }

    segments.append(segment)
    save_segments(segments)

    return {
        "status": "created",
        "segment": segment,
        "audio_url": f"/api/v1/audio/{filename}",
    }


@router.get("/segments")
async def list_segments():
    return {"segments": load_segments()}


@router.post("/segments/update")
async def update_segment(payload: SegmentUpdateRequest):
    segments = load_segments()

    for seg in segments:
        if seg.get("id") == payload.id:
            old_text = normalize_text(str(seg.get("text", "") or ""))
            new_text = normalize_text(payload.text)

            if not new_text:
                raise HTTPException(status_code=400, detail="text empty")

            learning_raw = learn_from_text_edit(old_text, new_text)
            raw_changes = extract_learning_changes(learning_raw)
            filtered_changes = filter_learning_changes(raw_changes, min_confidence=0.50)
            learning = summarize_learning(learning_raw, filtered_changes)

            if filtered_changes:
                append_learning_log(
                    {
                        "segment_id": seg.get("id"),
                        "old_text": old_text,
                        "new_text": new_text,
                        "changes": filtered_changes,
                    }
                )

            path = SEGMENTS_DIR / seg["filename"]

            if path.exists():
                try:
                    path.unlink()
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"failed to replace audio: {exc}") from exc

            generate_audio(new_text, payload.voice_id, path)

            seg["text"] = new_text
            save_segments(segments)

            return {
                "status": "updated",
                "segment": seg,
                "audio_url": f"/api/v1/audio/{seg['filename']}",
                "learning": learning,
            }

    raise HTTPException(status_code=404, detail="segment not found")


@router.post("/segments/delete")
async def delete_segment(payload: SegmentDeleteRequest):
    segments = load_segments()
    new_segments: list[dict[str, Any]] = []
    deleted = None

    for s in segments:
        if s.get("id") == payload.id:
            deleted = s
        else:
            new_segments.append(s)

    if not deleted:
        raise HTTPException(status_code=404, detail="not found")

    path = SEGMENTS_DIR / deleted["filename"]

    if path.exists():
        try:
            path.unlink()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to delete file: {exc}") from exc

    save_segments(new_segments)

    return {"status": "deleted"}


# -----------------------------
# LEARNING
# -----------------------------
@router.post("/learn-from-edit")
async def learn_only(payload: LearnFromEditRequest):
    old_text = normalize_text(payload.old_text)
    new_text = normalize_text(payload.new_text)

    if not old_text or not new_text:
        raise HTTPException(status_code=400, detail="invalid input")

    learning_raw = learn_from_text_edit(old_text, new_text)
    raw_changes = extract_learning_changes(learning_raw)
    filtered_changes = filter_learning_changes(raw_changes, min_confidence=0.50)
    learning = summarize_learning(learning_raw, filtered_changes)

    if filtered_changes:
        append_learning_log(
            {
                "old_text": old_text,
                "new_text": new_text,
                "changes": filtered_changes,
            }
        )

    return {
        "status": "learned",
        "result": learning,
    }


@router.get("/learning-log")
async def get_learning_log():
    items = load_learning_log(limit=50)
    return {
        "items": items,
        "count": len(items),
    }


@router.post("/learning/accept")
async def accept_learning(payload: LearningDecisionRequest):
    original = normalize_text(payload.original)
    formatted = normalize_text(payload.formatted)
    category = payload.category or "misc_pronunciation"

    if not original or not formatted:
        raise HTTPException(status_code=400, detail="invalid input")

    if category not in DEFAULT_LEXICON:
        category = "misc_pronunciation"

    lexicon = load_lexicon()

    if not lexicon_entry_exists(lexicon, category, original, formatted):
        lexicon[category].append(
            {
                "original": original,
                "formatted": formatted,
            }
        )
        save_lexicon(lexicon)

    remove_learning_change(original, formatted)

    return {
        "status": "accepted",
        "original": original,
        "formatted": formatted,
        "category": category,
    }


@router.post("/learning/reject")
async def reject_learning(payload: LearningRejectRequest):
    original = normalize_text(payload.original)
    formatted = normalize_text(payload.formatted)

    if not original or not formatted:
        raise HTTPException(status_code=400, detail="invalid input")

    removed = remove_learning_change(original, formatted)

    return {
        "status": "rejected",
        "removed": removed,
        "original": original,
        "formatted": formatted,
    }


# -----------------------------
# LEXICON
# -----------------------------
@router.get("/lexicon")
async def get_lexicon():
    return {"lexicon": load_lexicon()}


@router.post("/lexicon/add")
async def add_lexicon_word(
    original: str,
    formatted: str,
    category: str = "misc_pronunciation",
):
    original = normalize_text(original)
    formatted = normalize_text(formatted)

    if not original or not formatted:
        raise HTTPException(status_code=400, detail="invalid input")

    lexicon = load_lexicon()

    if category not in lexicon:
        raise HTTPException(status_code=400, detail="invalid category")

    if lexicon_entry_exists(lexicon, category, original):
        raise HTTPException(status_code=400, detail="word already exists")

    lexicon[category].append(
        {
            "original": original,
            "formatted": formatted,
        }
    )

    save_lexicon(lexicon)
    return {"status": "added"}


@router.post("/lexicon/update")
async def update_lexicon_word(
    original: str,
    formatted: str,
    category: str = "misc_pronunciation",
):
    original = normalize_text(original)
    formatted = normalize_text(formatted)

    if not original or not formatted:
        raise HTTPException(status_code=400, detail="invalid input")

    lexicon = load_lexicon()

    if category not in lexicon:
        raise HTTPException(status_code=400, detail="invalid category")

    for item in lexicon[category]:
        if isinstance(item, dict) and normalize_text(str(item.get("original", "") or "")) == original:
            item["formatted"] = formatted
            save_lexicon(lexicon)
            return {"status": "updated"}

    raise HTTPException(status_code=404, detail="word not found")


@router.post("/lexicon/delete")
async def delete_lexicon_word(
    original: str,
    category: str = "misc_pronunciation",
):
    original = normalize_text(original)

    if not original:
        raise HTTPException(status_code=400, detail="invalid input")

    lexicon = load_lexicon()

    if category not in lexicon:
        raise HTTPException(status_code=400, detail="invalid category")

    before = len(lexicon[category])
    lexicon[category] = [
        item for item in lexicon[category]
        if not (isinstance(item, dict) and normalize_text(str(item.get("original", "") or "")) == original)
    ]

    if len(lexicon[category]) == before:
        raise HTTPException(status_code=404, detail="word not found")

    save_lexicon(lexicon)
    return {"status": "deleted"}


@router.get("/lexicon/preview")
async def preview_lexicon_word(text: str, voice: str = "salem_podcast"):
    text = normalize_text(text)

    if not text:
        raise HTTPException(status_code=400, detail="text empty")

    audio_path = tts_manager.generate_temp_audio(
        text=prepare_tts_text(text),
        voice_id=voice,
    )

    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=500, detail="preview failed")

    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename="preview.wav",
    )


@router.post("/lexicon/learn")
async def learn_lexicon(payload: LexiconLearnRequest):
    original = normalize_text(payload.original)
    edited = normalize_text(payload.edited)
    category = payload.category or "misc_pronunciation"

    if not original or not edited:
        raise HTTPException(status_code=400, detail="invalid input")

    learning_raw = learn_from_text_edit(original, edited)
    raw_changes = extract_learning_changes(learning_raw)
    filtered_changes = filter_learning_changes(raw_changes, min_confidence=0.50)

    if not filtered_changes:
        return {
            "status": "ignored",
            "saved": 0,
            "changes": [],
        }

    if category not in DEFAULT_LEXICON:
        category = "misc_pronunciation"

    saved_count = add_changes_to_lexicon(filtered_changes, default_category=category)

    append_learning_log(
        {
            "source": "lexicon_learn",
            "old_text": original,
            "new_text": edited,
            "changes": filtered_changes,
            "saved_to_lexicon": saved_count,
        }
    )

    return {
        "status": "learned",
        "saved": saved_count,
        "changes": filtered_changes,
    }


# -----------------------------
# AUDIO
# -----------------------------
@router.get("/audio/{filename}")
async def get_audio(filename: str):
    path = SEGMENTS_DIR / filename

    if not path.exists():
        raise HTTPException(status_code=404, detail="file not found")

    return FileResponse(path, media_type="audio/wav")