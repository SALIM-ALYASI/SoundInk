from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import lfilter
from TTS.api import TTS
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range

from core.voice_registry import get_voice


PAUSE_MARKERS = {"$", "$$"}

# مستوى الذروة المستهدف عند تطبيع مرجع الصوت قبل إرساله لـXTTS.
# مرجع هادئ الصوت (peak منخفض) يخلي الصوت المستنسخ يميل لبحّة/خشونة.
REFERENCE_TARGET_PEAK = 0.9

# ── إعدادات "الماسترنق" (Mastering) للصوت الناتج ──
# مستوى بروز خفيف حول 3kHz يزيد وضوح الحروف (خصوصًا الصفير والحروف الحادة)،
# نفس الفكرة المستخدمة بالتعليق الصوتي والإعلانات المذاعة.
EQ_PRESENCE_FREQ_HZ = 3000
EQ_PRESENCE_GAIN_DB = 3.0
EQ_PRESENCE_Q = 1.0

# ضغط ديناميكي خفيف يخلي مستوى الصوت ثابت بدل ما يتفاوت بين جزء وجزء.
COMPRESSOR_THRESHOLD_DB = -20.0
COMPRESSOR_RATIO = 3.0
COMPRESSOR_ATTACK_MS = 5.0
COMPRESSOR_RELEASE_MS = 60.0

# معيار البث الشائع للإعلانات/البودكاست (LUFS).
TARGET_LOUDNESS_LUFS = -16.0

# سقف أمان لمنع القطع الرقمي (clipping) بعد رفع مستوى الصوت.
SAFETY_PEAK_CEILING = 0.97


class InferenceManager:
    def __init__(self) -> None:
        self._tts: Optional[TTS] = None
        self._normalized_ref_cache: dict[str, str] = {}

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
        """
        تغيير سرعة النطق بدون المساس بطبقة الصوت (pitch).

        الطريقة القديمة كانت تغيّر معدل العينات (frame_rate) مباشرة، وهذا
        يرفع/يخفض طبقة الصوت مع السرعة (نفس أثر تسريع أو إبطاء شريط كاسيت)،
        وهو ما كان يُنتج صوتًا مجهدًا/أجش يشبه البحّة خصوصًا عند أطراف
        النطاق المسموح به (0.8x - 1.3x). نستخدم بدلها phase vocoder عبر
        librosa اللي يغيّر المدة فقط ويحافظ على طبقة الصوت الطبيعية.
        """
        clean_speed = self._normalize_speed(speed)

        if abs(clean_speed - 1.0) < 0.01:
            return audio_path

        y, sr = librosa.load(audio_path, sr=None, mono=True)
        stretched = librosa.effects.time_stretch(y, rate=clean_speed)
        sf.write(audio_path, stretched, sr)

        return audio_path

    def _apply_presence_eq(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        فلتر EQ من نوع peaking (صيغة RBJ Audio EQ Cookbook القياسية) يرفع
        منطقة الوضوح (~3kHz) شوي، يخلي الصوت أوضح وأقرب لصوت "الراديو/التعليق
        الصوتي" المستخدم بالإعلانات، بدون ما يأثر على باقي الطيف الترددي.
        """
        gain_db = EQ_PRESENCE_GAIN_DB
        freq = EQ_PRESENCE_FREQ_HZ
        q = EQ_PRESENCE_Q

        a_gain = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * freq / sr
        alpha = np.sin(w0) / (2 * q)
        cos_w0 = np.cos(w0)

        b0 = 1 + alpha * a_gain
        b1 = -2 * cos_w0
        b2 = 1 - alpha * a_gain
        a0 = 1 + alpha / a_gain
        a1 = -2 * cos_w0
        a2 = 1 - alpha / a_gain

        b = np.array([b0, b1, b2]) / a0
        a = np.array([a0, a1, a2]) / a0

        return lfilter(b, a, y).astype(np.float32)

    def _apply_compression(self, audio_path: str) -> None:
        """
        ضغط ديناميكي خفيف عبر pydub — يخلي مستوى الصوت ثابت طوال المقطع
        بدل ما يتفاوت، صفة أساسية بأي تسجيل صوتي "منتَج" احترافيًا.
        """
        sound = AudioSegment.from_file(audio_path)

        compressed = compress_dynamic_range(
            sound,
            threshold=COMPRESSOR_THRESHOLD_DB,
            ratio=COMPRESSOR_RATIO,
            attack=COMPRESSOR_ATTACK_MS,
            release=COMPRESSOR_RELEASE_MS,
        )

        compressed.export(audio_path, format="wav")

    def _apply_loudness_normalization(self, audio_path: str) -> None:
        """
        تطبيع مستوى الصوت لمعيار بث ثابت (LUFS) بدل الاعتماد على الذروة فقط،
        نفس الأسلوب المستخدم بمنصات البودكاست/الإعلانات لضمان صوت متسق
        الحجم بين مقطع وآخر. مع سقف أمان لمنع القطع الرقمي (clipping).
        """
        y, sr = librosa.load(audio_path, sr=None, mono=True)

        meter = pyln.Meter(sr)
        current_loudness = meter.integrated_loudness(y)

        if current_loudness == float("-inf"):
            return

        normalized = pyln.normalize.loudness(y, current_loudness, TARGET_LOUDNESS_LUFS)

        peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
        if peak > SAFETY_PEAK_CEILING:
            normalized = normalized * (SAFETY_PEAK_CEILING / peak)

        sf.write(audio_path, normalized, sr)

    def _master_audio(self, audio_path: str) -> str:
        """
        سلسلة معالجة الصوت النهائية بعد التوليد مباشرة: وضوح (EQ) ← ضغط
        ديناميكي ← تطبيع الحجم الصوتي. الهدف صوت أقرب لجودة الإعلانات
        المذاعة الاحترافية بدل صوت خام مباشر من نموذج الاستنساخ الصوتي.
        """
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        y = self._apply_presence_eq(y, sr)

        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > SAFETY_PEAK_CEILING:
            y = y * (SAFETY_PEAK_CEILING / peak)

        sf.write(audio_path, y, sr)

        self._apply_compression(audio_path)
        self._apply_loudness_normalization(audio_path)

        return audio_path

    def _normalize_reference_audio(self, ref_path: str) -> str:
        """
        تطبيع مستوى صوت ملف المرجع قبل إرساله لنموذج الاستنساخ الصوتي.

        ملفات مرجعية مسجّلة بمستوى منخفض (peak ضعيف) تخلي النموذج يستنسخ
        صوتًا أقرب للهمس/البحّة. نرفع مستوى الذروة لمستوى موحّد بدون
        المساس بالملف الأصلي (نكتب نسخة مؤقتة فقط، ونخزّنها بذاكرة مؤقتة
        عشان ما نعيد التطبيع لنفس الصوت بكل مقطع).
        """
        cached = self._normalized_ref_cache.get(ref_path)
        if cached and os.path.exists(cached):
            return cached

        y, sr = librosa.load(ref_path, sr=None, mono=True)

        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak <= 0.0:
            return ref_path

        gain = REFERENCE_TARGET_PEAK / peak
        # ما نطبّق كسب أقل من 1 إلا لو الملف فعلاً أعلى من الهدف،
        # ولا نرفع الكسب لدرجة تفقد فيها معنى (clipping) محمي أصلاً بالقسمة على الذروة.
        normalized = np.clip(y * gain, -1.0, 1.0)

        temp_path = self._create_temp_output_path()
        sf.write(temp_path, normalized, sr)

        self._normalized_ref_cache[ref_path] = temp_path
        return temp_path

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

        reference_wav = self._normalize_reference_audio(str(Path(voice["ref_wav"])))

        tts.tts_to_file(
            text=clean_text,
            speaker_wav=reference_wav,
            language=voice["language"],
            file_path=output_path,
        )

        self._postprocess_speed(output_path, clean_speed)
        self._master_audio(output_path)

        return output_path

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