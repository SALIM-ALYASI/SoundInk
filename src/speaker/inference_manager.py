from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from TTS.api import TTS

from core.voice_registry import get_voice


PAUSE_MARKERS = {"$", "$$"}


class InferenceManager:
    def __init__(self) -> None:
        self._tts: Optional[TTS] = None

    def _load_model(self) -> TTS:
        if self._tts is None:
            print("Loading XTTS model...")
            self._tts = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2"
            )
            print("XTTS model loaded.")
        return self._tts

    def _create_temp_output_path(self) -> str:
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        return temp_path

    def _normalize_input_text(self, text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            raise ValueError("Input text is empty.")
        return cleaned

    def _is_pause_marker(self, text: str) -> bool:
        return (text or "").strip() in PAUSE_MARKERS

    def _resolve_voice(self, voice_id: str | None = None) -> dict:
        voice = get_voice(voice_id)

        ref_wav = str(voice.get("ref_wav", "")).strip()
        if not voice.get("exists") or not ref_wav:
            raise FileNotFoundError(
                f"Voice reference not found: {ref_wav or 'missing ref_wav'}"
            )

        ref_path = Path(ref_wav)
        if not ref_path.exists():
            raise FileNotFoundError(f"Voice file does not exist: {ref_path}")

        language = str(voice.get("language", "")).strip()
        if not language:
            raise ValueError("Voice language is missing.")

        return voice

    def _synthesize_to_file(
        self,
        text: str,
        voice: dict,
        output_path: str,
    ) -> str:
        clean_text = self._normalize_input_text(text)

        if self._is_pause_marker(clean_text):
            raise ValueError(
                f"Pause marker must not be sent to XTTS directly: {clean_text}"
            )

        preview = clean_text[:120] + ("..." if len(clean_text) > 120 else "")

        print("Generating audio...")
        print("Text preview:", preview)
        print("Using voice:", voice.get("id", "unknown"))
        print("Reference wav:", voice.get("ref_wav"))

        tts = self._load_model()
        tts.tts_to_file(
            text=clean_text,
            speaker_wav=str(Path(voice["ref_wav"])),
            language=voice["language"],
            file_path=output_path,
        )

        return output_path

    def generate_temp_audio(
        self,
        text: str,
        voice_id: str | None = None,
    ) -> str:
        clean_text = self._normalize_input_text(text)

        if self._is_pause_marker(clean_text):
            return clean_text

        voice = self._resolve_voice(voice_id)
        temp_path = self._create_temp_output_path()

        return self._synthesize_to_file(
            text=clean_text,
            voice=voice,
            output_path=temp_path,
        )

    def generate_temp_segments(
        self,
        segments: list[str],
        voice_id: str | None = None,
    ) -> list[str]:
        if not segments:
            raise ValueError("Segments list is empty.")

        normalized_segments = [
            (segment or "").strip()
            for segment in segments
            if (segment or "").strip()
        ]

        if not normalized_segments:
            raise ValueError("No valid segments to synthesize.")

        voice = self._resolve_voice(voice_id)
        output_paths: list[str] = []

        print("Total segments:", len(normalized_segments))

        for index, segment in enumerate(normalized_segments, start=1):
            print(f"Segment {index}/{len(normalized_segments)}")

            if self._is_pause_marker(segment):
                print(f"Detected pause marker: {segment}")
                output_paths.append(segment)
                continue

            temp_path = self._create_temp_output_path()

            self._synthesize_to_file(
                text=segment,
                voice=voice,
                output_path=temp_path,
            )

            output_paths.append(temp_path)

        return output_paths


tts_manager = InferenceManager()