from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from core.audio_merger import merge_wav_files, mix_voice_with_bgm_ducking

router = APIRouter()

DATA_DIR = Path("data")
SEGMENTS_FILE = DATA_DIR / "segments.json"
SEGMENTS_AUDIO_DIR = Path("outputs/segments")
PRESET_AUDIO_DIR = DATA_DIR / "preset_audio"
FINAL_OUTPUT_DIR = Path("outputs/final")
BGM_DIR = Path("assets") / "bgm"

FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INTRO_FILE = PRESET_AUDIO_DIR / "intro.wav"
OUTRO_FILE = PRESET_AUDIO_DIR / "outro.wav"
SILENCE_FILE = PRESET_AUDIO_DIR / "silence.wav"


class MergeSegmentsRequest(BaseModel):
    segment_ids: list[str] = Field(..., min_length=1)
    silence_ms: int = 3000
    output_name: str = "merged_segments"
    delete_segments_after_merge: bool = True


class MergeFullEpisodeRequest(BaseModel):
    segment_ids: list[str] = Field(..., min_length=1)
    include_intro: bool = True
    include_outro: bool = True
    include_silence: bool = True
    silence_ms: int = 3000
    output_name: str = "full_episode"
    delete_segments_after_merge: bool = True


class MergeEpisodeWithBgmRequest(BaseModel):
    segment_ids: list[str] = Field(..., min_length=1)
    include_intro: bool = True
    include_outro: bool = True
    include_silence: bool = True
    silence_ms: int = 3000
    bgm_id: str = Field(..., min_length=1)
    output_name: str = "full_episode_bgm"
    delete_segments_after_merge: bool = True


def load_segments() -> list[dict[str, Any]]:
    if not SEGMENTS_FILE.exists():
        return []

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


def find_segment_by_id(segment_id: str) -> dict[str, Any] | None:
    for segment in load_segments():
        if segment.get("id") == segment_id:
            return segment
    return None


def resolve_segment_audio_path(segment_id: str) -> Path:
    segment = find_segment_by_id(segment_id)

    if not segment:
        raise HTTPException(
            status_code=404,
            detail=f"المقطع غير موجود: {segment_id}",
        )

    filename = segment.get("filename")
    if not filename:
        raise HTTPException(
            status_code=400,
            detail=f"المقطع بلا ملف صوتي: {segment_id}",
        )

    audio_path = SEGMENTS_AUDIO_DIR / filename

    if not audio_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"ملف الصوت غير موجود للمقطع: {segment_id}",
        )

    return audio_path


def ensure_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"ملف {label} غير موجود: {path}",
        )


def find_bgm_file(bgm_id: str) -> Path | None:
    if not bgm_id:
        return None

    for ext in (".wav", ".mp3"):
        candidate = BGM_DIR / f"{bgm_id}{ext}"
        if candidate.exists():
            return candidate

    return None


def safe_output_name(name: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in ("_", "-", ".") else "_"
        for ch in (name or "").strip()
    )
    cleaned = cleaned.strip("._")
    return cleaned or "output"


def build_merge_inputs(
    segment_ids: list[str],
    silence_ms: int,
    include_intro: bool = False,
    include_outro: bool = False,
    include_silence: bool = True,
) -> list[Any]:
    merge_inputs: list[Any] = []

    if include_intro:
        ensure_file_exists(INTRO_FILE, "المقدمة")
        merge_inputs.append(str(INTRO_FILE))

        if include_silence:
            merge_inputs.append({"path": None, "silence_ms": silence_ms})

    for index, segment_id in enumerate(segment_ids):
        segment_audio_path = resolve_segment_audio_path(segment_id)
        merge_inputs.append(str(segment_audio_path))

        is_last = index == len(segment_ids) - 1
        if include_silence and not is_last:
            merge_inputs.append({"path": None, "silence_ms": silence_ms})

    if include_outro:
        ensure_file_exists(OUTRO_FILE, "الخاتمة")

        if include_silence and len(segment_ids) > 0:
            merge_inputs.append({"path": None, "silence_ms": silence_ms})

        merge_inputs.append(str(OUTRO_FILE))

    return merge_inputs


def export_merged_to_final(
    merge_inputs: list[Any],
    output_name: str,
) -> Path:
    try:
        merged_temp_path = merge_wav_files(
            chunks=merge_inputs,
            silence_between_segments_ms=0,
            silence_between_paragraphs_ms=0,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"فشل دمج الصوت: {exc}",
        ) from exc

    final_path = FINAL_OUTPUT_DIR / f"{safe_output_name(output_name)}.wav"

    try:
        shutil.copyfile(merged_temp_path, final_path)
    finally:
        try:
            Path(merged_temp_path).unlink(missing_ok=True)
        except Exception:
            pass

    return final_path


def apply_bgm_to_file(voice_path: Path, bgm_id: str, output_name: str) -> Path:
    bgm_path = find_bgm_file(bgm_id)

    if not bgm_path:
        raise HTTPException(
            status_code=404,
            detail=f"ملف الموسيقى غير موجود: {bgm_id}",
        )

    try:
        mixed_temp_path = mix_voice_with_bgm_ducking(
            voice_path=str(voice_path),
            bgm_path=str(bgm_path),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"فشل دمج الموسيقى الخلفية: {exc}",
        ) from exc

    final_path = FINAL_OUTPUT_DIR / f"{safe_output_name(output_name)}.wav"

    try:
        shutil.copyfile(mixed_temp_path, final_path)
    finally:
        try:
            Path(mixed_temp_path).unlink(missing_ok=True)
        except Exception:
            pass

    return final_path


def delete_used_segments(segment_ids: list[str]) -> None:
    segments = load_segments()
    remaining_segments: list[dict[str, Any]] = []

    for segment in segments:
        seg_id = segment.get("id")
        filename = segment.get("filename")

        if seg_id in segment_ids:
            if filename:
                audio_path = SEGMENTS_AUDIO_DIR / filename
                if audio_path.exists():
                    try:
                        audio_path.unlink()
                    except Exception:
                        pass
        else:
            remaining_segments.append(segment)

    save_segments(remaining_segments)


@router.post("/merge/segments-only")
async def merge_segments_only(payload: MergeSegmentsRequest):
    merge_inputs = build_merge_inputs(
        segment_ids=payload.segment_ids,
        silence_ms=payload.silence_ms,
        include_intro=False,
        include_outro=False,
        include_silence=True,
    )

    final_path = export_merged_to_final(
        merge_inputs=merge_inputs,
        output_name=payload.output_name,
    )

    if payload.delete_segments_after_merge:
        delete_used_segments(payload.segment_ids)

    return FileResponse(
        path=str(final_path),
        media_type="audio/wav",
        filename=final_path.name,
    )


@router.post("/merge/full-episode")
async def merge_full_episode(payload: MergeFullEpisodeRequest):
    merge_inputs = build_merge_inputs(
        segment_ids=payload.segment_ids,
        silence_ms=payload.silence_ms,
        include_intro=payload.include_intro,
        include_outro=payload.include_outro,
        include_silence=payload.include_silence,
    )

    final_path = export_merged_to_final(
        merge_inputs=merge_inputs,
        output_name=payload.output_name,
    )

    if payload.delete_segments_after_merge:
        delete_used_segments(payload.segment_ids)

    return FileResponse(
        path=str(final_path),
        media_type="audio/wav",
        filename=final_path.name,
    )


@router.post("/merge/full-episode-with-bgm")
async def merge_full_episode_with_bgm(payload: MergeEpisodeWithBgmRequest):
    merge_inputs = build_merge_inputs(
        segment_ids=payload.segment_ids,
        silence_ms=payload.silence_ms,
        include_intro=payload.include_intro,
        include_outro=payload.include_outro,
        include_silence=payload.include_silence,
    )

    voice_only_path = export_merged_to_final(
        merge_inputs=merge_inputs,
        output_name=f"{payload.output_name}_voice_only",
    )

    final_path = apply_bgm_to_file(
        voice_path=voice_only_path,
        bgm_id=payload.bgm_id,
        output_name=payload.output_name,
    )

    try:
        voice_only_path.unlink(missing_ok=True)
    except Exception:
        pass

    if payload.delete_segments_after_merge:
        delete_used_segments(payload.segment_ids)

    return FileResponse(
        path=str(final_path),
        media_type="audio/wav",
        filename=final_path.name,
    )