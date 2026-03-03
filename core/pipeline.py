import uuid
from typing import List, Dict, Any
from pathlib import Path

from core.logger import setup_logger
from core.echo_core import EchoCore
from core.echo_persona import PersonaManager
from core.echo_renderer import AudioRenderer
from core.session_manager import SessionManager
from core.config import settings

logger = setup_logger("EchoPipeline")

class EchoPipeline:
    """
    Orchestrates the massive Unified Dual-Output Pipeline
    and supports Segment Self-Healing and Cleanup.
    """
    def __init__(self, tts_manager=None):
        self.session_manager = SessionManager()
        self.persona = PersonaManager()
        self.core = EchoCore(tts_manager=tts_manager, persona_manager=self.persona)
        self.renderer = AudioRenderer()
        
        # We can dynamically decide background music from config later, 
        # but for now we hardcode the cinematic asset as per Phase 1 logic.
        self.default_bgm = settings.BASE_DIR / "echo_agent" / "assets" / "music" / "calm_music.ogg"

    def split_into_segments(self, text: str) -> List[str]:
        """Splits long text into manageable sentences for self-healing chunking."""
        # Simple logical split on punctuation or newlines.
        # Can be enhanced later via analyzer.py logic if complex parsing is needed.
        parts = text.replace('\n', ' ').split('.')
        segments = [p.strip() for p in parts if p.strip()]
        return segments

    def generate_dual_output(self, text: str, voice_id: str = "voice1", bgm_id: str = None) -> Dict[str, Any]:
        """
        1. Creates a Session.
        2. Splits Text.
        3. Generates chunks.
        4. Stitches into Normal MP3.
        5. Ducks with BGM -> Cinematic MP3.
        """
        session_id = str(uuid.uuid4())
        session_dir = self.session_manager.create_session(session_id)
        tmp_dir = self.session_manager.get_temp_dir(session_id)
        
        logger.info(f"Starting Unified Pipeline for Session: {session_id}")
        
        # Create dict representation for EchoCore
        parsed_segments = [{"speaker": voice_id, "text": chunk} for chunk in self.split_into_segments(text)]
        
        if not parsed_segments:
            return {"error": "Text segmenting yielded no content."}

        # Step 1: Generate Chunks (Normal)
        logger.info("Generating raw voice chunks...")
        chunk_paths = self.core.generate_audio_from_segments(parsed_segments, tmp_dir)
        
        # Step 2: Stitch Normal Output
        normal_wav = tmp_dir / "normal_merged.wav"
        normal_mp3 = session_dir / "normal.mp3"
        
        # --- Smart Cinematic Pacing Algorithm ---
        import random
        # Base silence before any audio starts to let the BGM establish mood
        intro_pause = round(random.uniform(2.0, 3.5), 2)
        
        pauses = []
        for i in range(len(chunk_paths)):
            if i == len(chunk_paths) - 1:
                # Outro: long cinematic fadeout space (3 to 5 seconds)
                pauses.append(round(random.uniform(3.0, 5.0), 2))
            else:
                # Inter-segment: Smart dynamic dramatic pauses based on text chunk length
                segment_len = len(parsed_segments[i]["text"])
                if segment_len > 80:
                    pauses.append(round(random.uniform(1.2, 2.0), 2)) # Longer pause for long sentences
                else:
                    pauses.append(round(random.uniform(0.6, 1.2), 2)) # Shorter pause for short sentences
        
        logger.info("Stitching chunks into Normal WAV with Smart Pacing...")
        self.renderer.concatenate_audio(chunk_paths, pauses, normal_wav, tmp_dir, intro_pause=intro_pause)
        
        logger.info("Converting Normal to MP3...")
        self.renderer.convert_to_mp3(normal_wav, normal_mp3)

        # Step 3: Mix Cinematic Output
        cinematic_mp3 = session_dir / "cinematic.mp3"
        
        # Determine BGM Path
        bgm_path = self.default_bgm
        if bgm_id:
            user_bgm = Path("/Users/alyasi/Downloads/mp3") / bgm_id
            if user_bgm.exists():
                bgm_path = user_bgm
        
        logger.info(f"Mixing Cinematic Output with BGM & Ducking using {bgm_path}...")
        self.renderer.add_background_music(
            speech_path=normal_wav, 
            bgm_path=bgm_path, 
            output_path=cinematic_mp3,
            bgm_volume=0.20,
            duck_ratio=10.0
        )
        
        # Save Metadata for self-healing state
        meta = {
            "session_id": session_id,
            "voice_id": voice_id,
            "bgm_id": bgm_id,
            "segments": parsed_segments,
            "chunk_files": [p.name for p in chunk_paths],
            "pauses": pauses,
            "intro_pause": intro_pause
        }
        
        import json
        (session_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        
        return {
            "session_id": session_id,
            "normal_url": f"/api/v1/session/{session_id}/normal.mp3",
            "cinematic_url": f"/api/v1/session/{session_id}/cinematic.mp3",
            "segments": parsed_segments
        }

    def heal_segment(self, session_id: str, segment_index: int, new_text: str) -> Dict[str, Any]:
        """
        Replaces ONE chunk without recreating the whole audio file.
        1. Deletes old chunk.
        2. Regenerates new chunk.
        3. Restitches Normal & Cinematic.
        """
        session_dir = self.session_manager.get_session_dir(session_id)
        tmp_dir = self.session_manager.get_temp_dir(session_id)
        meta_file = session_dir / "meta.json"
        
        if not meta_file.exists():
            return {"error": "Session metadata not found. Cannot heal."}
            
        import json
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        
        if segment_index < 0 or segment_index >= len(meta["segments"]):
            return {"error": "Invalid segment index."}
            
        voice_id = meta["voice_id"]
        meta["segments"][segment_index]["text"] = new_text
        
        logger.info(f"Healing Segment {segment_index} for Session {session_id}...")
        
        # Generate just the fixed segment
        chunk_name = meta["chunk_files"][segment_index]
        chunk_path = tmp_dir / chunk_name
        
        if chunk_path.exists():
            chunk_path.unlink() # Delete bad audio
            
        # Re-generate it
        self.core.process_single_text(new_text, chunk_path, speaker_id=voice_id)
        
        # Re-stitch
        chunk_paths = [tmp_dir / name for name in meta["chunk_files"]]
        normal_wav = tmp_dir / "normal_merged.wav"
        normal_mp3 = session_dir / "normal.mp3"
        cinematic_mp3 = session_dir / "cinematic.mp3"
        
        intro_pause = meta.get("intro_pause", 0.0)
        
        if normal_wav.exists():
            normal_wav.unlink()
            
        self.renderer.concatenate_audio(chunk_paths, meta["pauses"], normal_wav, tmp_dir, intro_pause=intro_pause)
        self.renderer.convert_to_mp3(normal_wav, normal_mp3)
        
        bgm_id = meta.get("bgm_id")
        bgm_path = self.default_bgm
        if bgm_id:
            user_bgm = Path("/Users/alyasi/Downloads/mp3") / bgm_id
            if user_bgm.exists():
                bgm_path = user_bgm
        
        self.renderer.add_background_music(
            speech_path=normal_wav, 
            bgm_path=bgm_path, 
            output_path=cinematic_mp3,
            bgm_volume=0.20,
            duck_ratio=10.0
        )
        
        # Save updated text
        meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        
        return {
            "status": "success",
            "message": f"Segment {segment_index} regenerated successfully.",
            "session_id": session_id,
            "normal_url": f"/api/v1/session/{session_id}/normal.mp3",
            "cinematic_url": f"/api/v1/session/{session_id}/cinematic.mp3",
            "updated_text": new_text
        }
