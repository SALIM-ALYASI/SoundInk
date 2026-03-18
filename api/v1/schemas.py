from typing import Optional

from pydantic import BaseModel, Field


# =====================================================
# 🎧 Speak (Preview)
# =====================================================
class SpeakRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description=(
            "النص المراد تحويله إلى صوت للمعاينة. "
            "يمكن استخدام $ و $$ داخل النص كعلامات توقف، "
            "لكن تأثيرها الكامل يظهر في مسار البودكاست."
        ),
    )

    voice_id: str = Field(
        default="salem_podcast",
        min_length=1,
        description="معرف الصوت المستخدم",
    )

    preview: bool = Field(
        default=True,
        description="وضع المعاينة السريعة",
    )


# =====================================================
# ✂️ Segmentation Preview
# =====================================================
class SegmentRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description=(
            "النص الذي سيتم تقسيمه. "
            "يمكن وضع $ كسطر مستقل لإضافة توقف 10 ثوانٍ، "
            "و$$ كسطر مستقل لإضافة توقف 15 ثانية."
        ),
    )

    voice_id: str = Field(
        default="salem_podcast",
        min_length=1,
        description="الصوت المستخدم في المعاينة",
    )


# =====================================================
# 🎙️ Podcast Generation
# =====================================================
class PodcastRequest(BaseModel):
    # ------------------------
    # 📄 Text Input
    # ------------------------
    text: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description=(
            "النص الكامل للحلقة. "
            "يدعم علامات التوقف داخل النص: "
            "$ = 10 ثواني صمت، $$ = 15 ثانية صمت. "
            "ويُفضّل وضعها كسطر مستقل أو فقرة مستقلة."
        ),
    )

    episode_title: Optional[str] = Field(
        default="podcast_episode",
        max_length=120,
        description="اسم الحلقة أو اسم الملف النهائي",
    )

    # ------------------------
    # 🎧 Voice Settings
    # ------------------------
    voice_id: str = Field(
        default="salem_podcast",
        min_length=1,
        description="الصوت المستخدم في الحلقة",
    )

    style: Optional[str] = Field(
        default=None,
        max_length=50,
        description="نمط الأداء المستقبلي مثل calm أو energetic أو deep",
    )

    # ------------------------
    # 🎼 Background Music
    # ------------------------
    bgm_id: Optional[str] = Field(
        default="echowave",
        max_length=120,
        description="معرف موسيقى الخلفية",
    )

    # ------------------------
    # ⏱️ Timing Controls
    # ------------------------
    silence_between_segments_ms: int = Field(
        default=500,
        ge=0,
        le=5000,
        description="مدة الصمت بين الجمل بالملي ثانية",
    )

    silence_between_paragraphs_ms: int = Field(
        default=1400,
        ge=0,
        le=8000,
        description="مدة الصمت بين الفقرات بالملي ثانية",
    )

    intro_lead_ms: int = Field(
        default=2000,
        ge=0,
        le=8000,
        description="مدة الصمت قبل بداية الحلقة بالملي ثانية",
    )

    skip_marker_pause_ms: int = Field(
        default=1500,
        ge=0,
        le=8000,
        description="مدة الوقفة عند العناوين التنظيمية مثل المقدمة والفقرة الأولى",
    )

    # ------------------------
    # ⚡ Execution Mode
    # ------------------------
    fast_mode: bool = Field(
        default=True,
        description="تشغيل المسار الأساسي بدون retry لتحسين الاستقرار",
    )