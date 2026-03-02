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
from core.models import User, GenerationHistory
from api.v1.auth import get_current_user

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
    return {
        "voices": [
            {"id": "voice1", "name": "Salem Base (Neutral)"},
            {"id": "voice2", "name": "Deep Cinema (Trailer)"}
        ]
    }

@router.post("/generate")
async def generate_audio(
    request: Request, 
    body: SynthesizeRequest, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Starts the dual-output generation asynchronously via Celery Queue. Consumes user tokens."""
    # Apply Rate Limiting
    check_rate_limit(request, max_requests=10, time_window_sec=60)
    
    char_count = len(body.text)
    
    # Validation: Token check
    if current_user.token_balance < char_count:
        raise HTTPException(
            status_code=402, 
            detail=f"Insufficient tokens. Request needs {char_count} but you have {current_user.token_balance} left."
        )
        
    # Deduct wallet
    current_user.token_balance -= char_count
    
    # Enqueue task to Redis/Celery
    task = generate_audio_task.delay(body.text, body.voice_id)
    
    # Create History Record (status pending basically, linked via task.id as session_id for now)
    history_record = GenerationHistory(
        session_id=task.id,
        user_id=current_user.id,
        voice_id=body.voice_id,
        characters_used=char_count
    )
    db.add(history_record)
    db.commit()
    
    return {"message": "Generation started", "task_id": task.id, "tokens_remaining": current_user.token_balance}

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
    body: LexiconRequest,
    current_user: User = Depends(get_current_user)
):
    """Add a permanent phonetics bypass into lexicon memory. Requires Login."""
    check_rate_limit(request, max_requests=30, time_window_sec=60)
    lexicon.add_correction(body.original, body.corrected)
    return {"error": False, "message": f"Added rule: {body.original} -> {body.corrected}"}

@router.get("/history")
async def get_user_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Returns a list of past generation sessions for the dashboard."""
    history = db.query(GenerationHistory).filter(GenerationHistory.user_id == current_user.id).order_by(GenerationHistory.created_at.desc()).all()
    return {"history": history}

@router.get("/session/{session_id}/{filename}")
async def fetch_media(session_id: str, filename: str):
    """Retrieve rendered audio (normal or cinematic)."""
    session_dir = pipeline.session_manager.get_session_dir(session_id)
    audio_path = session_dir / filename
    
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
        
    mime = "audio/mpeg" if filename.endswith(".mp3") else "audio/wav"
    return FileResponse(audio_path, media_type=mime, headers={"Cache-Control": "max-age=3600"})
