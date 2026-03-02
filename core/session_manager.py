import shutil
import time
from pathlib import Path
from core.config import settings
from core.logger import setup_logger

logger = setup_logger("SessionManager")

class SessionManager:
    """Manages generation isolated sessions and temp file cleanup."""
    def __init__(self):
        self.sessions_dir = settings.STORAGE_DIR / "sessions"
        self.tmp_dir = settings.STORAGE_DIR / "tmp"
        
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self, session_id: str) -> Path:
        """Creates a dedicated directory for a generation session."""
        session_path = self.sessions_dir / session_id
        session_path.mkdir(parents=True, exist_ok=True)
        return session_path

    def get_session_dir(self, session_id: str) -> Path:
        """Returns the session directory path."""
        return self.sessions_dir / session_id
        
    def session_exists(self, session_id: str) -> bool:
        return self.get_session_dir(session_id).exists()

    def get_temp_dir(self, task_id: str) -> Path:
        tmp = self.tmp_dir / task_id
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Deletes sessions older than max_age_hours."""
        now = time.time()
        count = 0
        for session_dir in self.sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue
            
            dir_age = now - session_dir.stat().st_mtime
            if dir_age > (max_age_hours * 3600):
                try:
                    shutil.rmtree(session_dir)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to delete old session {session_dir.name}: {e}")
                    
        if count > 0:
            logger.info(f"Cleaned up {count} old sessions.")

    def cleanup_temp_files(self, max_age_hours: int = 2):
        """Aggressively deletes temporary files."""
        now = time.time()
        for tmp_item in self.tmp_dir.iterdir():
            age = now - tmp_item.stat().st_mtime
            if age > (max_age_hours * 3600):
                try:
                    if tmp_item.is_dir():
                        shutil.rmtree(tmp_item)
                    else:
                        tmp_item.unlink()
                except Exception as e:
                    logger.error(f"Failed to delete temp file {tmp_item.name}: {e}")
