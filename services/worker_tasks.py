import json
import uuid
import time
from pathlib import Path
from typing import Dict, Any

from celery import shared_task
from core.logger import setup_logger
from core.pipeline import EchoPipeline
from core.session_manager import SessionManager

# Initialize components locally within the worker process
try:
    from src.speaker.inference_manager import tts_manager
except ImportError:
    tts_manager = None

pipeline = EchoPipeline(tts_manager=tts_manager)
session_manager = SessionManager()
logger = setup_logger("CeleryWorker")

@shared_task(bind=True, name="services.worker_tasks.generate_audio_task")
def generate_audio_task(self, text: str, voice_id: str = "voice1") -> Dict[str, Any]:
    """
    Heavy lifting: Synthesizes text to dual-output chunks in the background.
    """
    logger.info(f"Worker started Task {self.request.id} for voice {voice_id}")
    self.update_state(state="PROCESSING", meta={'progress': 'Starting Generation'})
    
    try:
        start_time = time.time()
        result = pipeline.generate_dual_output(text, voice_id)
        
        if "error" in result:
            logger.error(f"Task {self.request.id} failed: {result['error']}")
            return {"status": "error", "message": result["error"]}
        
        elapsed = round(time.time() - start_time, 2)
        logger.info(f"Task {self.request.id} completed successfully in {elapsed}s")
        
        return {
            "status": "completed",
            "session_id": result["session_id"],
            "normal_url": result["normal_url"],
            "cinematic_url": result["cinematic_url"],
            "segments": result["segments"]
        }
    except Exception as e:
        logger.exception(f"Exception in Task {self.request.id}: {str(e)}")
        return {"status": "error", "message": f"Worker Exception: {str(e)}"}


@shared_task(bind=True, name="services.worker_tasks.heal_segment_task")
def heal_segment_task(self, session_id: str, segment_index: int, new_text: str) -> Dict[str, Any]:
    """
    Background job to heal a specific segment to prevent blocking API.
    """
    logger.info(f"Worker healing segment {segment_index} for Session {session_id}")
    self.update_state(state="PROCESSING", meta={'progress': 'Healing Segment'})
    
    try:
        result = pipeline.heal_segment(session_id, segment_index, new_text)
        if "error" in result:
            return {"status": "error", "message": result["error"]}
        
        return result
    except Exception as e:
        logger.exception(f"Exception during healing: {e}")
        return {"status": "error", "message": str(e)}


@shared_task(name="services.worker_tasks.scheduled_cleanup_task")
def scheduled_cleanup_task():
    """
    Celery Beat task: periodically cleans up temporary output folders.
    """
    logger.info("Executing scheduled cleanup task...")
    try:
        # Delete unused tmp files older than 2 hours
        session_manager.cleanup_temp_files(max_age_hours=2)
        # Delete full sessions older than 48 hours to save disk space
        session_manager.cleanup_old_sessions(max_age_hours=48)
        logger.info("Cleanup successful.")
    except Exception as e:
        logger.error(f"Cleanup task failed: {str(e)}")
