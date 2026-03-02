import os
from typing import List, Dict, Any
from pathlib import Path
from core.echo_persona import PersonaManager
from core.logger import setup_logger
from core.config import settings

logger = setup_logger("EchoCore")

class EchoCore:
    """
    Core orchestrator for text-to-speech generation.
    Takes scripts, segments them (if needed), applies persona rules,
    and directly triggers RAM-based TTS inference.
    """
    def __init__(self, tts_manager=None, persona_manager: PersonaManager = None):
        self.tts_manager = tts_manager
        self.persona = persona_manager or PersonaManager()
        
        # We try to import InferenceManager directly if not passed in
        if not self.tts_manager:
            try:
                from src.speaker.inference_manager import tts_manager as fallback_manager
                self.tts_manager = fallback_manager
                logger.info("Imported InferenceManager successfully in EchoCore.")
            except ImportError as e:
                logger.warning(f"Could not automatically load InferenceManager: {e}")

    def generate_audio_from_segments(self, segments: List[Dict[str, Any]], output_dir: Path) -> List[Path]:
        """
        Takes a list of pre-parsed speaker segments (e.g. from analyzer.py)
        and generates WAV loops for each.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_paths = []

        if not self.tts_manager:
            raise RuntimeError("TTS Manager is not initialized. Cannot generate audio.")

        for i, seg in enumerate(segments):
            speaker_id = seg.get("speaker", "voice1")
            raw_text = seg.get("text", "")
            
            # 1. Apply Persona Text Filtering (Lexicon + Cleanup)
            processed_text = self.persona.format_text_pipeline(raw_text, use_chatty=True)
            
            if not processed_text.strip():
                logger.warning(f"Segment {i} was empty after processing. Skipping.")
                continue

            # 2. Define Output Paths
            out_name = f"seg_{i:04d}_{speaker_id}.wav"
            out_path = output_dir / out_name

            # 3. Resolve Reference Audio
            # Note: In Phase 2.5 we can load dynamic refs from config.
            # For now, default fallback to salem_podcast_clean.wav if we don't have a map
            ref_wav = str(settings.BASE_DIR / "data" / "ref" / "salem_podcast_clean.wav")
            
            if speaker_id == "voice2":
                alt_ref = settings.BASE_DIR / "data" / "ref" / "ref_master.wav"
                if alt_ref.exists():
                    ref_wav = str(alt_ref)

            # 4. Generate
            logger.info(f"Generating segment {i} for {speaker_id} -> {out_name}")
            try:
                self.tts_manager.generate_audio(
                    text=processed_text,
                    output_path=str(out_path),
                    ref_wav=ref_wav
                )
                generated_paths.append(out_path)
            except Exception as e:
                logger.error(f"Failed to generate segment {i}: {e}")
                raise

        return generated_paths

    def process_single_text(self, text: str, output_path: Path, speaker_id: str = "voice1") -> Path:
        """
        Utility to generate a single block of text without segment mapping.
        """
        temp_seg = [{"speaker": speaker_id, "text": text}]
        out_dir = output_path.parent
        paths = self.generate_audio_from_segments(temp_seg, out_dir)
        
        if paths:
            # Rename the chunk to the requested output path
            if paths[0] != output_path:
                paths[0].rename(output_path)
            return output_path
        else:
            raise RuntimeError("Audio generation yielded no output.")
