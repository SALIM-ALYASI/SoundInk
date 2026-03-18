from __future__ import annotations

from pathlib import Path
from functools import lru_cache

try:
    import whisper
except ImportError:
    whisper = None


DEFAULT_WHISPER_MODEL = "base"


@lru_cache(maxsize=2)
def get_whisper_model(model_name: str = DEFAULT_WHISPER_MODEL):
    if whisper is None:
        raise ImportError(
            "whisper is not installed. Install it with: pip install openai-whisper"
        )

    return whisper.load_model(model_name)


def transcribe_audio(
    audio_path: str,
    model_name: str = DEFAULT_WHISPER_MODEL,
    language: str = "ar",
) -> str:
    """
    Local ASR using Whisper.

    Args:
        audio_path: path to wav/mp3 audio
        model_name: whisper model name, e.g. tiny, base, small
        language: transcription language, default Arabic

    Returns:
        recognized text or empty string on failure
    """
    path = Path(audio_path)

    if not path.exists():
        return ""

    try:
        model = get_whisper_model(model_name)
        result = model.transcribe(
            str(path),
            language=language,
            fp16=False,
            verbose=False,
        )

        text = (result.get("text") or "").strip()
        return text

    except Exception:
        return ""