import uuid
from typing import Dict, Any
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from pydantic import ValidationError

from api.v1.schemas import SynthesizeRequest, HealRequest, LexiconRequest
from api.rate_limiter import check_rate_limit
from core.pipeline import EchoPipeline
from core.lexicon_memory import LexiconMemory
from services.worker_tasks import generate_audio_task, heal_segment_task
from celery.result import AsyncResult
from core.celery_app import celery_app

from sqlalchemy.orm import Session
from fastapi import Depends
from core.database import get_db
from core.config import settings
from pathlib import Path as _Path
from core.models import GenerationHistory

BGM_DIR = settings.BASE_DIR / "assets" / "bgm"

# API Router setup
router = APIRouter()
pipeline = EchoPipeline(tts_manager=None) # Only needed for file manager logic inside API now
lexicon = LexiconMemory()

@router.get("/health")
async def health_check():
    """Verify engine and TTS status."""
    return {
        "status": "up",
        "engine": "Echo Audio Engine Phase 4",
        "queue_broker": celery_app.conf.broker_url
    }

@router.get("/voices")
async def list_voices():
    """Returns available base voice IDs."""
    import os
    from pathlib import Path
    
    ref_dir = Path(__file__).resolve().parents[2] / "data" / "ref"
    voices = []
    
    if ref_dir.exists():
        for file in ref_dir.glob("*.wav"):
            # Provide friendly names for known files
            if file.name == "salem_base.wav":
                name = "سالم (النبرة القديمة)"
            elif file.name == "salem_podcast_clean.wav":
                name = "الياسي (النبرة الجديدة)"
            elif file.name == "female_soft.wav":
                name = "ابتسام (نبرة ناعمة وجديدة)"
            else:
                name = file.stem.replace("_", " ").title()
                
            voices.append({"id": file.name, "name": name})
            
    # Inject dialogue option at the top of the list
    voices.insert(0, {"id": "dialogue", "name": "حوار (متعدد الأصوات)"})
            
    if len(voices) == 1: # if only dialogue is there
        voices.append({"id": "salem_podcast_clean.wav", "name": "Salem Podcast (Default)"})
        
    return {"voices": voices}

@router.get("/voices/{voice_id}/preview")
async def preview_voice(voice_id: str):
    """Serve the raw reference audio file for UI preview."""
    import os
    from pathlib import Path
    
    ref_path = Path(__file__).resolve().parents[2] / "data" / "ref" / voice_id
    if not ref_path.exists() or not str(ref_path).endswith('.wav'):
        raise HTTPException(status_code=404, detail="Voice preview not found")
        
    return FileResponse(ref_path, media_type="audio/wav")

@router.get("/bgms")
async def list_bgms():
    """Returns available background music files."""
    bgm_dir = BGM_DIR
    bgms = []
    if bgm_dir.exists():
        for file in bgm_dir.glob("*.mp3"):
            bgms.append({"id": file.name, "name": file.name})
    return {"bgms": bgms}

@router.get("/bgms/{bgm_id}/preview")
async def preview_bgm(bgm_id: str):
    """Serve the raw BGM audio file for UI preview."""
    bgm_path = BGM_DIR / bgm_id
    if not bgm_path.exists() or not str(bgm_path).endswith('.mp3'):
        raise HTTPException(status_code=404, detail="BGM preview not found")
        
    return FileResponse(bgm_path, media_type="audio/mpeg")

@router.post("/generate")
async def generate_audio(
    request: Request, 
    body: SynthesizeRequest, 
    db: Session = Depends(get_db)
):
    """Starts the dual-output generation asynchronously via Celery Queue."""
    # Apply Rate Limiting
    check_rate_limit(request, max_requests=10, time_window_sec=60)
    
    char_count = len(body.text)
    
    # Enqueue task to Redis/Celery
    task = generate_audio_task.delay(body.text, body.voice_id, body.bgm_id, body.mode)
    
    # Create History Record (anonymous)
    history_record = GenerationHistory(
        session_id=task.id,
        user_id="anonymous",
        voice_id=body.voice_id,
        characters_used=char_count
    )
    db.add(history_record)
    db.commit()
    
    return {"message": "Generation started", "task_id": task.id}

@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Poll for background completion returning media links from Celery AsyncResult."""
    res = AsyncResult(task_id, app=celery_app)
    
    if res.state == 'PENDING':
        return {"status": "processing", "message": "Task is queued."}
    elif res.state == 'PROCESSING':
        return {"status": "processing", "message": res.info.get('progress', '') if res.info else ''}
    elif res.state == 'SUCCESS':
        data = res.result
        if isinstance(data, dict) and data.get("status") == "error":
            raise HTTPException(status_code=400, detail=data.get("message", "Task failed internally"))
        return data
    elif res.state == 'FAILURE':
        raise HTTPException(status_code=500, detail=str(res.info))
        
    return {"status": "processing", "state": res.state}

@router.post("/session/regenerate")
async def heal_audio_segment(request: Request, body: HealRequest):
    """Self-healing: Queues async task for healing a specific segment."""
    check_rate_limit(request, max_requests=25, time_window_sec=60)
    
    # Push to Celery
    task = heal_segment_task.delay(body.session_id, body.segment_index, body.new_text)
    return {"message": "Heal task started", "task_id": task.id}

@router.post("/lexicon")
async def update_lexicon(
    request: Request, 
    body: LexiconRequest
):
    """Add a permanent phonetics bypass into lexicon memory."""
    check_rate_limit(request, max_requests=30, time_window_sec=60)
    lexicon.add_correction(body.original, body.corrected)
    return {"error": False, "message": f"Added rule: {body.original} -> {body.corrected}"}

@router.get("/history")
async def get_user_history(db: Session = Depends(get_db)):
    """Returns a list of past generation sessions."""
    history = db.query(GenerationHistory).order_by(GenerationHistory.created_at.desc()).limit(50).all()
    return {"history": history}

@router.get("/session/{session_id}/{filename}")
async def fetch_media(session_id: str, filename: str, download: bool = False):
    """Retrieve rendered audio (normal or cinematic)."""
    session_dir = pipeline.session_manager.get_session_dir(session_id)
    audio_path = session_dir / filename
    
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
        
    mime = "audio/mpeg" if filename.endswith(".mp3") else "audio/wav"
    headers = {"Cache-Control": "max-age=3600"}
    if download:
        return FileResponse(audio_path, media_type=mime, headers=headers, filename=filename, content_disposition_type="attachment")
    return FileResponse(audio_path, media_type=mime, headers=headers)
