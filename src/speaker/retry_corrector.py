from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

from src.asr.transcriber import transcribe_audio
from src.lexicon.manager import create_suggestion
from src.speaker.inference_manager import tts_manager
from src.speaker.text_normalizer import prepare_tts_text


MIN_ACCEPTANCE_SCORE = 0.75
MAX_RETRIES = 2
MAX_SUGGESTION_WORDS = 3

FEEDBACK_FILE = Path(__file__).resolve().parents[2] / "data" / "lexicon" / "retry_feedback.jsonl"


@dataclass
class RetryResult:
    original_text: str
    prepared_text: str
    recognized_text: str
    audio_path: str
    similarity: float
    accepted: bool
    attempts: int
    error: str = ""


def calc_similarity(a: str, b: str) -> float:
    a = (a or "").strip()
    b = (b or "").strip()

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def word_count(text: str) -> int:
    cleaned = (text or "").strip()
    if not cleaned:
        return 0
    return len(cleaned.split())


def should_create_suggestion(
    original_text: str,
    prepared_text: str,
    recognized_text: str,
) -> bool:
    original_words = word_count(original_text)
    prepared_words = word_count(prepared_text)
    recognized_words = word_count(recognized_text)

    max_words = max(original_words, prepared_words, recognized_words)

    if max_words == 0:
        return False

    if max_words > MAX_SUGGESTION_WORDS:
        return False

    return True


# 🧠 تعلم من الصوت (ASR)
def learn_from_asr(
    *,
    original_text: str,
    prepared_text: str,
    recognized_text: str,
    similarity: float,
    audio_path: str,
) -> None:

    if not should_create_suggestion(
        original_text=original_text,
        prepared_text=prepared_text,
        recognized_text=recognized_text,
    ):
        return

    if not recognized_text.strip():
        return

    if similarity >= 0.92:
        return

    # فرق بسيط → تجاهل
    if abs(len(prepared_text) - len(recognized_text)) < 2 and similarity > 0.85:
        return

    try:
        create_suggestion(
            original=original_text,
            suggested=recognized_text,  # 🔥 يعتمد على الصوت الحقيقي
            recognized=recognized_text,
            similarity=similarity,
            audio_path=audio_path,
        )
    except Exception as exc:
        print(f"ASR learning skipped: {exc}")


def log_retry_feedback(
    original_text: str,
    prepared_text: str,
    recognized_text: str,
    similarity: float,
    accepted: bool,
    attempts: int,
    voice_id: str,
    audio_path: str = "",
    error: str = "",
) -> None:
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": utc_now_iso(),
        "original": original_text,
        "prepared": prepared_text,
        "recognized": recognized_text,
        "similarity": round(similarity, 4),
        "accepted": accepted,
        "attempts": attempts,
        "voice_id": voice_id,
        "audio_path": audio_path,
        "error": error,
    }

    try:
        with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"Failed to write retry feedback: {exc}")


def build_retry_text(original_text: str, recognized_text: str) -> str:
    _ = recognized_text
    return prepare_tts_text(original_text)


def try_create_suggestion(
    *,
    original_text: str,
    prepared_text: str,
    recognized_text: str,
    similarity: float,
    audio_path: str,
) -> None:

    if not should_create_suggestion(
        original_text=original_text,
        prepared_text=prepared_text,
        recognized_text=recognized_text,
    ):
        return

    try:
        create_suggestion(
            original=original_text,
            suggested=prepared_text,
            recognized=recognized_text,
            similarity=similarity,
            audio_path=audio_path,
        )
    except Exception as exc:
        print(f"Suggestion creation skipped: {exc}")


def generate_and_validate_segment(
    text: str,
    voice_id: str,
    min_score: float = MIN_ACCEPTANCE_SCORE,
    max_retries: int = MAX_RETRIES,
) -> RetryResult:

    original_text = text or ""
    prepared_text = prepare_tts_text(original_text)

    best_audio_path: str = ""
    best_recognized_text = ""
    best_similarity = -1.0
    best_prepared_text = prepared_text
    best_error = ""
    attempts = 0

    current_text = prepared_text

    for attempt in range(1, max_retries + 2):
        attempts = attempt

        try:
            audio_path = tts_manager.generate_temp_audio(
                text=current_text,
                voice_id=voice_id,
            )
        except Exception as exc:
            error_msg = f"TTS generation failed: {exc}"

            log_retry_feedback(
                original_text=original_text,
                prepared_text=current_text,
                recognized_text="",
                similarity=0.0,
                accepted=False,
                attempts=attempts,
                voice_id=voice_id,
                audio_path="",
                error=error_msg,
            )

            return RetryResult(
                original_text=original_text,
                prepared_text=current_text,
                recognized_text="",
                audio_path="",
                similarity=0.0,
                accepted=False,
                attempts=attempts,
                error=error_msg,
            )

        try:
            recognized_text = transcribe_audio(audio_path) or ""
        except Exception as exc:
            recognized_text = ""
            error_msg = f"ASR transcription failed: {exc}"
        else:
            error_msg = ""

        similarity = calc_similarity(current_text, recognized_text)

        if similarity > best_similarity:
            best_similarity = similarity
            best_audio_path = audio_path
            best_recognized_text = recognized_text
            best_prepared_text = current_text
            best_error = error_msg

        if similarity >= min_score:

            log_retry_feedback(
                original_text=original_text,
                prepared_text=current_text,
                recognized_text=recognized_text,
                similarity=similarity,
                accepted=True,
                attempts=attempts,
                voice_id=voice_id,
                audio_path=audio_path,
                error=error_msg,
            )

            # 🔥 تعلم من الصوت
            learn_from_asr(
                original_text=original_text,
                prepared_text=current_text,
                recognized_text=recognized_text,
                similarity=similarity,
                audio_path=audio_path,
            )

            # fallback
            try_create_suggestion(
                original_text=original_text,
                prepared_text=current_text,
                recognized_text=recognized_text,
                similarity=similarity,
                audio_path=audio_path,
            )

            return RetryResult(
                original_text=original_text,
                prepared_text=current_text,
                recognized_text=recognized_text,
                audio_path=audio_path,
                similarity=similarity,
                accepted=True,
                attempts=attempts,
                error=error_msg,
            )

        current_text = build_retry_text(original_text, recognized_text)

    final_similarity = max(best_similarity, 0.0)

    # 🔥 تعلم من الفشل
    learn_from_asr(
        original_text=original_text,
        prepared_text=best_prepared_text,
        recognized_text=best_recognized_text,
        similarity=final_similarity,
        audio_path=best_audio_path,
    )

    try_create_suggestion(
        original_text=original_text,
        prepared_text=best_prepared_text,
        recognized_text=best_recognized_text,
        similarity=final_similarity,
        audio_path=best_audio_path,
    )

    log_retry_feedback(
        original_text=original_text,
        prepared_text=best_prepared_text,
        recognized_text=best_recognized_text,
        similarity=final_similarity,
        accepted=False,
        attempts=attempts,
        voice_id=voice_id,
        audio_path=best_audio_path,
        error=best_error,
    )

    return RetryResult(
        original_text=original_text,
        prepared_text=best_prepared_text,
        recognized_text=best_recognized_text,
        audio_path=best_audio_path,
        similarity=final_similarity,
        accepted=False,
        attempts=attempts,
        error=best_error,
    )