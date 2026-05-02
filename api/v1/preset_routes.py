from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pydub import AudioSegment

from src.speaker.inference_manager import tts_manager
from src.speaker.text_normalizer import prepare_tts_text

router = APIRouter()

PRESET_DIR = Path("data/preset_audio")
PRESET_DIR.mkdir(parents=True, exist_ok=True)

INTRO_TEXT = """
أهلاً بكم…
هذا بودكاست رَسيس مع سالم الحَجري،
حيث نقترب من الأفكار… ونترك أثرًا يبقى.
"""

OUTRO_TEXT = """
وبين كل فكرةٍ وأخرى…
يبقى الأثر.

… هذا كان بودكاست رَسيس
إلى لقاءٍ قريب.
"""


class PresetBuildRequest(BaseModel):
    voice_id: str = "salem_podcast"
    intro_speed: float = Field(default=0.96, ge=0.8, le=1.3)
    outro_speed: float = Field(default=1.0, ge=0.8, le=1.3)
    silence_ms: int = Field(default=3000, ge=0, le=10000)


def generate_audio(
    text: str,
    voice_id: str,
    output_path: Path,
    speed: float = 1.0,
):
    clean_text = prepare_tts_text(text.strip())

    if not clean_text:
        raise HTTPException(status_code=400, detail="النص فارغ")

    try:
        temp_path = tts_manager.generate_temp_audio(
            text=clean_text,
            voice_id=voice_id,
            speed=speed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ TTS: {e}")

    if not temp_path or not Path(temp_path).exists():
        raise HTTPException(status_code=500, detail="فشل إنشاء الصوت")

    shutil.copyfile(temp_path, output_path)

    try:
        Path(temp_path).unlink(missing_ok=True)
    except Exception:
        pass


def generate_silence(duration_ms: int, output_path: Path):
    silence = AudioSegment.silent(duration=duration_ms)
    silence.export(output_path, format="wav")


@router.post("/preset/build")
async def build_presets(payload: PresetBuildRequest):
    intro_path = PRESET_DIR / "intro.wav"
    outro_path = PRESET_DIR / "outro.wav"
    silence_path = PRESET_DIR / "silence.wav"

    generate_audio(
        text=INTRO_TEXT,
        voice_id=payload.voice_id,
        output_path=intro_path,
        speed=payload.intro_speed,
    )

    generate_audio(
        text=OUTRO_TEXT,
        voice_id=payload.voice_id,
        output_path=outro_path,
        speed=payload.outro_speed,
    )

    generate_silence(payload.silence_ms, silence_path)

    return {
        "status": "ok",
        "files": {
            "intro": str(intro_path),
            "outro": str(outro_path),
            "silence": str(silence_path),
        },
        "settings": {
            "voice_id": payload.voice_id,
            "intro_speed": payload.intro_speed,
            "outro_speed": payload.outro_speed,
            "silence_ms": payload.silence_ms,
        },
    }


@router.get("/preset/{filename}")
async def get_preset(filename: str):
    allowed = {"intro.wav", "outro.wav", "silence.wav"}

    if filename not in allowed:
        raise HTTPException(status_code=404, detail="غير مسموح")

    path = PRESET_DIR / filename

    if not path.exists():
        raise HTTPException(status_code=404, detail="غير موجود")

    return FileResponse(
        str(path),
        media_type="audio/wav",
        filename=filename,
    )