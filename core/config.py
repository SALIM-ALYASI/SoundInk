from __future__ import annotations

import os
from functools import lru_cache

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    """
    إعدادات التطبيق المركزية، تُقرأ من متغيرات البيئة (.env).
    """

    def __init__(self) -> None:
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", "5050"))
        self.reload: bool = _get_bool("RELOAD", False)

        # مفتاح API اختياري لحماية نقاط الوصول من الاستخدام العام غير المصرّح.
        # لو تُرك فارغًا، تبقى الحماية معطّلة (مناسب للتشغيل المحلي فقط).
        self.api_key: str | None = os.getenv("API_KEY") or None

        # أصول CORS المسموحة، مفصولة بفواصل. افتراضيًا كل شي مسموح (تطوير محلي).
        cors_raw = os.getenv("CORS_ORIGINS", "*")
        self.cors_origins: list[str] = [
            origin.strip() for origin in cors_raw.split(",") if origin.strip()
        ]

        # اسم موديل XTTS، قابل للتغيير بدون تعديل الكود.
        self.tts_model_name: str = os.getenv(
            "TTS_MODEL_NAME",
            "tts_models/multilingual/multi-dataset/xtts_v2",
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
