import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

# Load the project base dir dynamically
BASE_DIR = Path(__file__).resolve().parents[2]

# Ensure we have the model path standard
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_REF = BASE_DIR / "data" / "ref" / "salem_podcast_clean.wav"

class InferenceManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        # Implement singleton pattern to ensure only one model exists in RAM
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(InferenceManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        # To avoid double initialization
        if self._initialized:
            return
            
        self._initialized = True
        self.tts = None
        
        # Lock specifically for generating audio since TTS APIs are often not thread-safe concurrently
        self.generate_lock = threading.Lock()
        print("InferenceManager initialized. Model is NOT loaded yet (Lazy Load).")

    def _load_model(self):
        """Loads the model into RAM if it hasn't been loaded already."""
        if self.tts is None:
            with self._lock:
                # Double check inside the lock
                if self.tts is None:
                    print("TTS Model is loading into RAM. This may take a moment...")
                    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
                    try:
                        from TTS.api import TTS
                        self.tts = TTS(MODEL_NAME)
                        print("TTS Model successfully loaded into RAM!")
                    except ImportError as e:
                        raise ImportError(f"TTS library not found. {e}")

    def generate_audio(self, text: str, output_path: str, ref_wav: str = str(DEFAULT_REF), language: str = "ar", **kwargs):
        """
        Generates audio for a given string of text. Thread-safe!
        kwargs can contain 'speaker_wav', etc., and we map 'ref_wav' to 'speaker_wav'.
        """
        self._load_model()
        
        # Ensure only one generation occurs at a time on the GPU/CPU to prevent OOM
        with self.generate_lock:
            # Reconstruct the text processing logic if necessary (clean, lexicon etc.)
            # For Phase 2 we simply replace subprocess invocation with direct execution
            from src.speaker.tts_cli import load_lexicon_map, apply_lexicon, clean_for_ar_tts, smart_filter_chatty, diacritize_selective
            
            # Apply standard cleaning pipeline directly
            # 1. Lexicon
            lex_file = BASE_DIR / "config" / "lexicon.json"
            lex_map = load_lexicon_map(lex_file)
            processed_text = apply_lexicon(text, lex_map)
            
            # 2. Arabic cleaning
            processed_text = clean_for_ar_tts(processed_text)
            
            # 3. Chatty
            processed_text = smart_filter_chatty(processed_text, starters=True)
            
            if not processed_text.strip():
                raise ValueError("Text was completely emptied after cleaning!")
                
            print(f"Generating audio to {output_path}...")
            
            # For simplicity, we just use a single invocation without split/concat since 
            # we want to ensure basic TTS works first directly in memory.
            # If split is needed for very long lines, we'd copy the split logic from tts_cli.
            # For now, let's keep it simple to replace the flask standard behavior.
            
            # We can handle long text safely using the split/concat logic from tts_cli but invoking the API directly natively.
            # For now, let's call the tts method!
            self.tts.tts_to_file(
                text=processed_text,
                file_path=str(output_path),
                speaker_wav=str(ref_wav),
                language=language
            )

            print(f"Generated directly from RAM to: {output_path}")

# Export a single global instance
tts_manager = InferenceManager()

# You can test locally via:
# if __name__ == "__main__":
#     manager = InferenceManager()
#     manager.generate_audio("مرحباً بك في عالم المعرفة", "test_inmem.wav")
