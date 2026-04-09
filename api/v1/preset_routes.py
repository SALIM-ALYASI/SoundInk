from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import shutil

from pydub import AudioSegment

from src.speaker.inference_manager import tts_manager
from src.speaker.text_normalizer import prepare_tts_text

router = APIRouter()

# 📁 مجلد التخزين
PRESET_DIR = Path("data/preset_audio")
PRESET_DIR.mkdir(parents=True, exist_ok=True)

# 🎙️ النصوص
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


# 🎧 توليد صوت
def generate_audio(text: str, voice_id: str, output_path: Path):
    clean_text = prepare_tts_text(text.strip())

    if not clean_text:
        raise HTTPException(status_code=400, detail="النص فارغ")

    try:
        temp_path = tts_manager.generate_temp_audio(
            text=clean_text,
            voice_id=voice_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ TTS: {e}")

    if not temp_path or not Path(temp_path).exists():
        raise HTTPException(status_code=500, detail="فشل إنشاء الصوت")

    shutil.copyfile(temp_path, output_path)

    try:
        Path(temp_path).unlink(missing_ok=True)
    except:
        pass


# 🔇 توليد صمت
def generate_silence(duration_ms: int, output_path: Path):
    silence = AudioSegment.silent(duration=duration_ms)
    silence.export(output_path, format="wav")


# 🚀 endpoint إنشاء الثلاثة
@router.post("/preset/build")
async def build_presets(voice_id: str = "salem_podcast"):
    intro_path = PRESET_DIR / "intro.wav"
    outro_path = PRESET_DIR / "outro.wav"
    silence_path = PRESET_DIR / "silence.wav"

    generate_audio(INTRO_TEXT, voice_id, intro_path)
    generate_audio(OUTRO_TEXT, voice_id, outro_path)
    generate_silence(3000, silence_path)

    return {
        "status": "ok",
        "files": {
            "intro": str(intro_path),
            "outro": str(outro_path),
            "silence": str(silence_path),
        }
    }


# 📥 تحميل الملفات
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
        filename=filename
    )