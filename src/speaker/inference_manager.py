from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from TTS.api import TTS
from pydub import AudioSegment

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

        if not isinstance(voice, dict):
            raise ValueError("Voice data is invalid.")

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

    def _normalize_speed(self, speed: float) -> float:
        try:
            value = float(speed)
        except (TypeError, ValueError):
            value = 1.0
        return max(0.8, min(value, 1.3))

    def _postprocess_speed(self, audio_path: str, speed: float) -> str:
        clean_speed = self._normalize_speed(speed)

        if abs(clean_speed - 1.0) < 0.01:
            return audio_path

        sound = AudioSegment.from_file(audio_path)

        modified = sound._spawn(
            sound.raw_data,
            overrides={"frame_rate": int(sound.frame_rate * clean_speed)}
        ).set_frame_rate(sound.frame_rate)

        modified.export(audio_path, format="wav")
        return audio_path

    def _synthesize_to_file(
        self,
        text: str,
        voice: dict,
        output_path: str,
        speed: float = 1.0,
    ) -> str:
        clean_text = self._normalize_input_text(text)

        if self._is_pause_marker(clean_text):
            raise ValueError(
                f"Pause marker must not be sent to XTTS directly: {clean_text}"
            )

        clean_speed = self._normalize_speed(speed)
        preview = clean_text[:120] + ("..." if len(clean_text) > 120 else "")

        print("Generating audio...")
        print("Text preview:", preview)
        print("Using voice:", voice.get("id", "unknown"))
        print("Reference wav:", voice.get("ref_wav"))
        print("Speed:", clean_speed)

        tts = self._load_model()

        tts.tts_to_file(
            text=clean_text,
            speaker_wav=str(Path(voice["ref_wav"])),
            language=voice["language"],
            file_path=output_path,
        )

        return self._postprocess_speed(output_path, clean_speed)

    def generate_temp_audio(
        self,
        text: str,
        voice_id: str | None = None,
        speed: float = 1.0,
    ) -> str:
        from src.speaker.segmenter import split_text_into_segments

        clean_text = self._normalize_input_text(text)

        if self._is_pause_marker(clean_text):
            return clean_text

        voice = self._resolve_voice(voice_id)
        segments = split_text_into_segments(clean_text, max_chars=140)

        print("Segments:", segments)

        audio_chunks: list[AudioSegment] = []
        temp_files: list[str] = []

        try:
            for seg in segments:
                seg = (seg or "").strip()
                if not seg:
                    continue

                if self._is_pause_marker(seg):
                    continue

                temp_path = self._create_temp_output_path()

                self._synthesize_to_file(
                    text=seg,
                    voice=voice,
                    output_path=temp_path,
                    speed=speed,
                )

                if os.path.exists(temp_path):
                    audio_chunks.append(AudioSegment.from_file(temp_path))
                    temp_files.append(temp_path)

            if not audio_chunks:
                raise RuntimeError("No audio generated")

            merged = AudioSegment.empty()

            for i, chunk in enumerate(audio_chunks):
                if i > 0:
                    merged += AudioSegment.silent(duration=180)
                merged += chunk

            final_path = self._create_temp_output_path()
            merged.export(final_path, format="wav")

            return final_path

        finally:
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception:
                    pass

    def generate_temp_segments(
        self,
        segments: list[str],
        voice_id: str | None = None,
        speed: float = 1.0,
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
        clean_speed = self._normalize_speed(speed)
        output_paths: list[str] = []

        print("Total segments:", len(normalized_segments))
        print("Speed:", clean_speed)

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
                speed=clean_speed,
            )

            output_paths.append(temp_path)

        return output_paths


tts_manager = InferenceManager()