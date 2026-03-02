import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from core.config import settings
from core.logger import setup_logger
from core.lexicon_memory import LexiconMemory

logger = setup_logger("EchoPersona")

class PersonaManager:
    """
    Manages voice identity, style, and lexicon context for the Echo Engine.
    Handles text pre-processing and dynamic overrides per persona.
    """
    def __init__(self, config_dir: Path = settings.CONFIG_DIR):
        self.config_dir = config_dir
        self.base_lexicon_map: Dict[str, str] = {}
        self.lexicon_memory = LexiconMemory()
        self.chatty_phrases = [
            ("لكن", "بس"), ("جدا", "مرّة"), ("كذلك", "برضه"), 
            ("أيضا", "بعد"), ("لذلك", "عشان كذا"), ("هذا الأمر", "هالموضوع"),
            ("يجب", "لازم"), ("ينبغي", "المفروض")
        ]
        self._load_global_lexicon()

    def _load_global_lexicon(self):
        """Loads the global terminology mapping from config/lexicon.json"""
        lex_file = self.config_dir / "lexicon.json"
        if not lex_file.exists():
            logger.warning(f"Lexicon file not found at {lex_file}")
            return
            
        try:
            data = json.loads(lex_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                # Simple K/V structure
                simple_pairs = {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}
                if len(simple_pairs) >= 1 and len(simple_pairs) == len(data):
                    self.base_lexicon_map = simple_pairs
                    return

                # Complex category structure
                for key in ["tribes_pronunciation", "names_pronunciation", "misc_pronunciation", "pronunciation"]:
                    arr = data.get(key)
                    if isinstance(arr, list):
                        for item in arr:
                            if isinstance(item, dict):
                                orig = item.get("original")
                                fmt = item.get("formatted")
                                if isinstance(orig, str) and isinstance(fmt, str) and orig.strip() and fmt.strip():
                                    self.base_lexicon_map[orig.strip()] = fmt.strip()
                                    
            logger.info(f"Loaded {len(self.base_lexicon_map)} base lexicon entries")
        except Exception as e:
            logger.error(f"Failed to load base lexicon: {e}")

    def apply_lexicon(self, text: str) -> str:
        """Replaces exact terminology from the merged lexicon map."""
        # Merge base static configs with interactive user fixes from LexiconMemory
        active_map = self.lexicon_memory.get_merged_lexicon(self.base_lexicon_map)
        
        if not active_map:
            return text

        # Replace larger strings first to prevent partial word collisions
        items = sorted(active_map.items(), key=lambda kv: len(kv[0]), reverse=True)
        boundary = r"[\s\.,،…:؛\{\}\"'«»]"

        def pat(word: str) -> str:
            w = re.escape(word)
            return rf"(?:(?<=^)|(?<={boundary})){w}(?:(?=$)|(?={boundary}))"

        for orig, fmt in items:
            text = re.sub(pat(orig), fmt, text)

        return text

    def apply_chatty_filter(self, text: str) -> str:
        """Applies a conversational Saudi-light style filter to formal text."""
        for a, b in self.chatty_phrases:
            text = text.replace(a, b)
        return text
        
    def clean_text_for_arabic_tts(self, text: str) -> str:
        """
        Removes non-arabic chars, parses digits to Arabic words, 
        and removes problematic symbols.
        """
        t = text.strip()
        
        # Emoji regex
        emoji_re = re.compile(
            "[\U0001F300-\U0001FAFF\u2600-\u26FF\u2700-\u27BF]+",
            flags=re.UNICODE
        )
        t = emoji_re.sub(" ", t)
        
        # Remove english chars and math symbols
        t = re.sub(r"[A-Za-z]+", " ", t)
        t = re.sub(r"[%@#_=+\/*\\<>[\]{}()|~^$`]", " ", t)
        t = t.replace("...", "…")
        t = re.sub(r"[؛;:]+", "،", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def format_text_pipeline(self, text: str, use_chatty: bool = True) -> str:
        """Runs the entire pre-TTS text normalization pipeline."""
        if not text:
            return ""
            
        t = self.apply_lexicon(text)
        if use_chatty:
            t = self.apply_chatty_filter(t)
        t = self.clean_text_for_arabic_tts(t)
        return t
