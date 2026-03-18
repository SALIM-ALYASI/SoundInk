from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import List, NotRequired, TypedDict, Union

from pydub import AudioSegment


PAUSE_MARKER_TO_MS = {
    "$": 10_000,
    "$$": 15_000,
}


class AudioChunk(TypedDict):
    path: str | None
    is_paragraph_break: bool
    silence_ms: NotRequired[int]


MergeInput = Union[str, AudioChunk]


def _validate_audio_file(path_str: str, label: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path_str}")
    return path


def _validate_non_negative(value: int | float, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} cannot be negative.")


def _load_audio(path_str: str, label: str) -> AudioSegment:
    path = _validate_audio_file(path_str, label)
    audio = AudioSegment.from_file(str(path))

    if len(audio) == 0:
        raise ValueError(f"{label} audio file is empty: {path_str}")

    return audio


def _export_temp_wav(audio: AudioSegment) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        output_path = temp_file.name

    audio.export(output_path, format="wav")
    return output_path


def _match_audio_profile(source: AudioSegment, reference: AudioSegment) -> AudioSegment:
    """
    توحيد خصائص الصوت حتى يكون الدمج والمكس أكثر استقرارًا.
    """
    audio = source

    if audio.channels != reference.channels:
        audio = audio.set_channels(reference.channels)

    if audio.sample_width != reference.sample_width:
        audio = audio.set_sample_width(reference.sample_width)

    if audio.frame_rate != reference.frame_rate:
        audio = audio.set_frame_rate(reference.frame_rate)

    return audio


def _loop_to_duration(audio: AudioSegment, target_duration_ms: int) -> AudioSegment:
    if target_duration_ms <= 0:
        raise ValueError("Target duration must be greater than zero.")

    if len(audio) == 0:
        raise ValueError("Cannot loop empty audio.")

    if len(audio) < target_duration_ms:
        repeat_count = math.ceil(target_duration_ms / len(audio))
        audio = audio * repeat_count

    return audio[:target_duration_ms]


def _safe_dbfs(audio: AudioSegment) -> float:
    """
    في الصمت الكامل قد تكون dBFS = -inf
    لذلك نعيد قيمة ثابتة منخفضة جدًا.
    """
    value = audio.dBFS
    if value == float("-inf"):
        return -120.0
    return value


def _build_silence(
    duration_ms: int,
    reference_profile: AudioSegment | None = None,
) -> AudioSegment:
    _validate_non_negative(duration_ms, "silence_ms")

    if reference_profile is not None:
        silence = AudioSegment.silent(
            duration=duration_ms,
            frame_rate=reference_profile.frame_rate,
        )

        if silence.channels != reference_profile.channels:
            silence = silence.set_channels(reference_profile.channels)

        if silence.sample_width != reference_profile.sample_width:
            silence = silence.set_sample_width(reference_profile.sample_width)

        return silence

    return AudioSegment.silent(duration=duration_ms)


def _resolve_chunk_pause(
    chunk: AudioChunk,
    silence_between_segments_ms: int,
    silence_between_paragraphs_ms: int,
) -> int:
    explicit_silence = chunk.get("silence_ms")
    if explicit_silence is not None:
        _validate_non_negative(int(explicit_silence), "silence_ms")
        return int(explicit_silence)

    return (
        silence_between_paragraphs_ms
        if bool(chunk.get("is_paragraph_break", False))
        else silence_between_segments_ms
    )


def _marker_to_chunk(marker: str) -> AudioChunk:
    silence_ms = PAUSE_MARKER_TO_MS.get(marker)
    if silence_ms is None:
        raise ValueError(f"Unsupported pause marker: {marker}")

    return {
        "path": None,
        "is_paragraph_break": False,
        "silence_ms": silence_ms,
    }


def _normalize_merge_inputs(chunks: List[MergeInput]) -> List[AudioChunk]:
    normalized: List[AudioChunk] = []

    for item in chunks:
        if isinstance(item, dict):
            normalized.append(
                {
                    "path": item.get("path"),
                    "is_paragraph_break": bool(item.get("is_paragraph_break", False)),
                    **(
                        {"silence_ms": int(item["silence_ms"])}
                        if item.get("silence_ms") is not None
                        else {}
                    ),
                }
            )
            continue

        if not isinstance(item, str):
            raise TypeError(f"Unsupported merge input type: {type(item).__name__}")

        stripped = item.strip()
        if not stripped:
            continue

        if stripped in PAUSE_MARKER_TO_MS:
            normalized.append(_marker_to_chunk(stripped))
            continue

        normalized.append(
            {
                "path": stripped,
                "is_paragraph_break": False,
            }
        )

    return normalized


def merge_wav_files(
    chunks: List[MergeInput],
    silence_between_segments_ms: int = 500,
    silence_between_paragraphs_ms: int = 1400,
) -> str:
    normalized_chunks = _normalize_merge_inputs(chunks)

    if not normalized_chunks:
        raise ValueError("No audio chunks provided for merge.")

    _validate_non_negative(
        silence_between_segments_ms,
        "silence_between_segments_ms",
    )
    _validate_non_negative(
        silence_between_paragraphs_ms,
        "silence_between_paragraphs_ms",
    )

    final_audio = AudioSegment.empty()
    reference_profile: AudioSegment | None = None
    has_real_audio = False

    for index, chunk in enumerate(normalized_chunks):
        file_path = chunk.get("path")

        # chunk صمت مخصص أو marker محول إلى صمت
        if not file_path:
            pause_ms = _resolve_chunk_pause(
                chunk=chunk,
                silence_between_segments_ms=silence_between_segments_ms,
                silence_between_paragraphs_ms=silence_between_paragraphs_ms,
            )
            final_audio += _build_silence(pause_ms, reference_profile)
            continue

        segment = _load_audio(file_path, "Chunk")

        if reference_profile is None:
            reference_profile = segment
        else:
            segment = _match_audio_profile(segment, reference_profile)

        final_audio += segment
        has_real_audio = True

        if index >= len(normalized_chunks) - 1:
            continue

        next_chunk = normalized_chunks[index + 1]

        # إذا التالي صمت مخصص أو marker، لا نضيف صمتًا تلقائيًا هنا
        if not next_chunk.get("path"):
            continue

        pause_ms = _resolve_chunk_pause(
            chunk=chunk,
            silence_between_segments_ms=silence_between_segments_ms,
            silence_between_paragraphs_ms=silence_between_paragraphs_ms,
        )

        if pause_ms > 0:
            final_audio += _build_silence(pause_ms, reference_profile)

    if not has_real_audio:
        raise ValueError("No real audio content found in chunks.")

    if len(final_audio) == 0:
        raise ValueError("Merged audio is empty.")

    return _export_temp_wav(final_audio)


def mix_voice_with_bgm(
    voice_path: str,
    bgm_path: str,
    bgm_gain_db: int = -18,
) -> str:
    voice_audio = _load_audio(voice_path, "Voice")
    bgm_audio = _load_audio(bgm_path, "BGM")

    bgm_audio = _match_audio_profile(bgm_audio, voice_audio)
    bgm_audio = _loop_to_duration(bgm_audio, len(voice_audio))
    bgm_audio = bgm_audio + bgm_gain_db

    mixed = bgm_audio.overlay(voice_audio)

    if len(mixed) == 0:
        raise ValueError("Mixed audio is empty.")

    return _export_temp_wav(mixed)


def mix_voice_with_bgm_ducking(
    voice_path: str,
    bgm_path: str,
    speaking_gain_db: int = -22,
    silent_gain_db: int = -12,
    window_ms: int = 300,
    silence_threshold_db: float = -38.0,
) -> str:
    if window_ms <= 0:
        raise ValueError("window_ms must be greater than zero.")

    voice_audio = _load_audio(voice_path, "Voice")
    bgm_audio = _load_audio(bgm_path, "BGM")

    bgm_audio = _match_audio_profile(bgm_audio, voice_audio)
    bgm_audio = _loop_to_duration(bgm_audio, len(voice_audio))

    dynamic_bgm = AudioSegment.empty()

    for start in range(0, len(voice_audio), window_ms):
        end = min(start + window_ms, len(voice_audio))

        voice_slice = voice_audio[start:end]
        bgm_slice = bgm_audio[start:end]

        voice_level = _safe_dbfs(voice_slice)
        gain_db = (
            speaking_gain_db
            if voice_level > silence_threshold_db
            else silent_gain_db
        )

        dynamic_bgm += bgm_slice + gain_db

    mixed = dynamic_bgm.overlay(voice_audio)

    if len(mixed) == 0:
        raise ValueError("Ducked mixed audio is empty.")

    return _export_temp_wav(mixed)