from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.speaker.inference_manager import tts_manager
from src.speaker.text_normalizer import prepare_tts_text

router = APIRouter()

JOBS_DIR = Path("data/tts_jobs")
OUTPUTS_DIR = Path("outputs/jobs")
JOBS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Prevent duplicate registration of the same deterministic job id and serialize
# XTTS inference on CPU. XTTS is intentionally run one job at a time.
_job_state_lock = threading.Lock()
_generation_lock = threading.Lock()

_SAFE_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class SpeakJobRequest(BaseModel):
    job_id: str = Field(..., min_length=1, max_length=128)
    text: str = Field(..., min_length=1, max_length=2000)
    voice_id: str | None = "salem_podcast"
    speed: float = Field(default=1.0, ge=0.8, le=1.3)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_job_id(job_id: str) -> str:
    value = (job_id or "").strip()
    if not _SAFE_JOB_ID_RE.fullmatch(value):
        raise HTTPException(
            status_code=400,
            detail="job_id must contain only letters, numbers, dot, underscore, or hyphen",
        )
    return value


def _metadata_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _audio_path(job_id: str) -> Path:
    return OUTPUTS_DIR / f"{job_id}.wav"


def _request_fingerprint(text: str, voice_id: str | None, speed: float) -> str:
    payload = {
        "text": " ".join((text or "").split()),
        "voice_id": voice_id or "salem_podcast",
        "speed": round(float(speed), 4),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_job(job_id: str) -> dict[str, Any] | None:
    path = _metadata_path(job_id)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _save_job(job: dict[str, Any]) -> None:
    path = _metadata_path(str(job["job_id"]))
    temp = path.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(job, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temp, path)


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    job_id = str(job["job_id"])
    result = dict(job)
    # Internal filesystem paths never need to leave the service.
    result.pop("audio_path", None)
    result["status_url"] = f"/api/v1/jobs/speak/{job_id}"
    result["audio_url"] = (
        f"/api/v1/jobs/speak/{job_id}/audio"
        if job.get("status") == "completed" and _audio_path(job_id).exists()
        else None
    )
    return result


def _run_job(job_id: str, text: str, voice_id: str | None, speed: float) -> None:
    temp_audio: str | None = None

    # One XTTS synthesis at a time on this CPU service. A submitted job remains
    # queued until it acquires this lock, but the HTTP request has already
    # returned 202 to n8n.
    with _generation_lock:
        job = _load_job(job_id)
        if not job or job.get("status") not in {"queued", "running"}:
            return

        job["status"] = "running"
        job["started_at"] = _now()
        job["updated_at"] = _now()
        _save_job(job)

        try:
            temp_audio = tts_manager.generate_temp_audio(
                text=prepare_tts_text(text),
                voice_id=voice_id or "salem_podcast",
                speed=speed,
            )

            if not temp_audio or not os.path.exists(temp_audio):
                raise RuntimeError("SoundInk did not produce an audio file")

            final_path = _audio_path(job_id)
            shutil.copyfile(temp_audio, final_path)

            job = _load_job(job_id) or job
            job["status"] = "completed"
            job["audio_path"] = str(final_path)
            job["audio_bytes"] = final_path.stat().st_size
            job["completed_at"] = _now()
            job["updated_at"] = _now()
            job["error"] = None
            _save_job(job)
        except Exception as exc:
            job = _load_job(job_id) or job
            job["status"] = "failed"
            job["failed_at"] = _now()
            job["updated_at"] = _now()
            job["error"] = str(exc)[:2000]
            _save_job(job)
        finally:
            if temp_audio and os.path.exists(temp_audio):
                try:
                    os.remove(temp_audio)
                except Exception:
                    pass


@router.post("/jobs/speak", status_code=202)
async def create_speak_job(
    payload: SpeakJobRequest,
    background_tasks: BackgroundTasks,
):
    job_id = _validate_job_id(payload.job_id)
    fingerprint = _request_fingerprint(
        payload.text,
        payload.voice_id,
        payload.speed,
    )

    with _job_state_lock:
        existing = _load_job(job_id)
        if existing:
            if existing.get("request_fingerprint") != fingerprint:
                raise HTTPException(
                    status_code=409,
                    detail="job_id already exists with different text, voice, or speed",
                )

            status = existing.get("status")
            if status in {"queued", "running", "completed"}:
                result = _public_job(existing)
                result["reused"] = True
                return result

            if status == "failed":
                raise HTTPException(
                    status_code=409,
                    detail="job failed previously; use the explicit retry endpoint",
                )

        now = _now()
        job: dict[str, Any] = {
            "job_id": job_id,
            "status": "queued",
            "voice_id": payload.voice_id or "salem_podcast",
            "speed": payload.speed,
            "request_fingerprint": fingerprint,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
            "failed_at": None,
            "audio_bytes": None,
            "audio_path": None,
            "error": None,
        }
        _save_job(job)

    background_tasks.add_task(
        _run_job,
        job_id,
        payload.text,
        payload.voice_id,
        payload.speed,
    )

    result = _public_job(job)
    result["reused"] = False
    return result


@router.get("/jobs/speak/{job_id}")
async def get_speak_job(job_id: str):
    job_id = _validate_job_id(job_id)
    job = _load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return _public_job(job)


@router.get("/jobs/speak/{job_id}/audio")
async def get_speak_job_audio(job_id: str):
    job_id = _validate_job_id(job_id)
    job = _load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail=f"job status is {job.get('status')}")

    path = _audio_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="audio file not found")

    return FileResponse(
        path,
        media_type="audio/wav",
        filename=f"{job_id}.wav",
    )


@router.post("/jobs/speak/{job_id}/retry", status_code=202)
async def retry_speak_job(
    job_id: str,
    payload: SpeakJobRequest,
    background_tasks: BackgroundTasks,
):
    job_id = _validate_job_id(job_id)
    if _validate_job_id(payload.job_id) != job_id:
        raise HTTPException(status_code=400, detail="payload job_id must match URL job_id")

    fingerprint = _request_fingerprint(
        payload.text,
        payload.voice_id,
        payload.speed,
    )

    with _job_state_lock:
        job = _load_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if job.get("request_fingerprint") != fingerprint:
            raise HTTPException(status_code=409, detail="retry payload does not match original job")
        if job.get("status") != "failed":
            raise HTTPException(
                status_code=409,
                detail=f"only failed jobs can be retried; current status is {job.get('status')}",
            )

        job["status"] = "queued"
        job["updated_at"] = _now()
        job["started_at"] = None
        job["completed_at"] = None
        job["failed_at"] = None
        job["audio_bytes"] = None
        job["audio_path"] = None
        job["error"] = None
        _save_job(job)

    background_tasks.add_task(
        _run_job,
        job_id,
        payload.text,
        payload.voice_id,
        payload.speed,
    )

    result = _public_job(job)
    result["reused"] = False
    result["retry"] = True
    return result


@router.delete("/jobs/speak/{job_id}")
async def delete_speak_job(job_id: str):
    job_id = _validate_job_id(job_id)

    with _job_state_lock:
        job = _load_job(job_id)
        if not job:
            return {"status": "not_found", "job_id": job_id}
        if job.get("status") in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="cannot delete a queued or running job")

        audio = _audio_path(job_id)
        metadata = _metadata_path(job_id)

        if audio.exists():
            audio.unlink()
        if metadata.exists():
            metadata.unlink()

    return {"status": "deleted", "job_id": job_id}
