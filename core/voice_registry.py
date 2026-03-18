from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_REF_DIR = PROJECT_ROOT / "data" / "ref"


VOICE_REGISTRY = {
    "salem_base": {
        "id": "salem_base",
        "label": "Salem Base",
        "ref_wav": DATA_REF_DIR / "salem_base.wav",
        "language": "ar",
        "active": True,
    },
    "salem_podcast": {
        "id": "salem_podcast",
        "label": "Salem Podcast",
        "ref_wav": DATA_REF_DIR / "salem_podcast_clean.wav",
        "language": "ar",
        "active": True,
    },
    "female_soft": {
        "id": "female_soft",
        "label": "Female Soft",
        "ref_wav": DATA_REF_DIR / "female_soft.wav",
        "language": "ar",
        "active": True,
    },
}

DEFAULT_VOICE_ID = "salem_podcast"


def list_voices() -> list[dict]:
    result = []
    for voice in VOICE_REGISTRY.values():
        item = dict(voice)
        item["ref_wav"] = str(item["ref_wav"])
        item["exists"] = Path(item["ref_wav"]).exists()
        result.append(item)
    return result


def get_voice(voice_id: str | None = None) -> dict:
    if voice_id and voice_id in VOICE_REGISTRY:
        voice = VOICE_REGISTRY[voice_id]
        if voice.get("active", False):
            item = dict(voice)
            item["ref_wav"] = str(item["ref_wav"])
            item["exists"] = Path(item["ref_wav"]).exists()
            return item

    voice = dict(VOICE_REGISTRY[DEFAULT_VOICE_ID])
    voice["ref_wav"] = str(voice["ref_wav"])
    voice["exists"] = Path(voice["ref_wav"]).exists()
    return voice