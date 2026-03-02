import json
import os
import time
from pathlib import Path
from typing import Dict, Any
from core.config import settings
from core.logger import setup_logger

logger = setup_logger("LexiconMemory")

class LexiconMemory:
    """
    Manages user-provided word corrections persistently.
    Corrections added here are prioritized over the global static lexicon.
    """
    def __init__(self):
        self.memory_dir = settings.STORAGE_DIR / "lexicon"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.memory_dir / "user_fixes.json"
        
        # Format: {"original_word": {"corrected_word": "...", "timestamp": 1234, "count": 1}}
        self.fixes: Dict[str, Dict[str, Any]] = {}
        self._load_memory()

    def _load_memory(self):
        if not self.memory_file.exists():
            return
        
        try:
            data = json.loads(self.memory_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.fixes = data
                logger.info(f"Loaded {len(self.fixes)} self-healing lexicon entries.")
        except Exception as e:
            logger.error(f"Failed to load user fixes: {e}")

    def _save_memory(self):
        try:
            self.memory_file.write_text(
                json.dumps(self.fixes, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"Failed to save user fixes: {e}")

    def add_correction(self, original_word: str, corrected_word: str):
        """Adds or updates a user-defined word correction."""
        original_word = original_word.strip()
        corrected_word = corrected_word.strip()
        
        if not original_word or not corrected_word:
            return

        if original_word in self.fixes:
            self.fixes[original_word]["corrected_word"] = corrected_word
            self.fixes[original_word]["timestamp"] = time.time()
            self.fixes[original_word]["count"] = self.fixes[original_word].get("count", 0) + 1
        else:
            self.fixes[original_word] = {
                "corrected_word": corrected_word,
                "timestamp": time.time(),
                "count": 1
            }
            
        self._save_memory()
        logger.info(f"Learned correction: '{original_word}' -> '{corrected_word}'")

    def get_merged_lexicon(self, base_lexicon: Dict[str, str]) -> Dict[str, str]:
        """Merges user fixes ON TOP of the base static lexicon map."""
        merged = base_lexicon.copy()
        for orig, data in self.fixes.items():
            merged[orig] = data["corrected_word"]
        return merged
