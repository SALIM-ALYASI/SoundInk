import os
import math
import logging
import random
from pathlib import Path
from pydub import AudioSegment
from pydub.generators import WhiteNoise
from google import genai
from pydantic import BaseModel
from typing import List, Optional

from src.speaker.inference_manager import InferenceManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParagraphPlan(BaseModel):
    text: str
    silence_before_sec: float
    reasoning: str

class AudioPlan(BaseModel):
    overall_bgm_file: str
    bgm_reasoning: str
    paragraphs: List[ParagraphPlan]

class CinematicProcessor:
    def __init__(self, bgm_dir: str, transitions_dir: Optional[str] = None):
        self.bgm_dir = Path(bgm_dir)
        self.transitions_dir = Path(transitions_dir) if transitions_dir else None
        self.tts_manager = InferenceManager()
        
        # Load available background music files
        self.available_bgm = [f.name for f in self.bgm_dir.glob("*.mp3")]
        if not self.available_bgm:
            logger.warning(f"No MP3 files found in {bgm_dir}")
        else:
            logger.info(f"Loaded {len(self.available_bgm)} background tracks.")
            
    def _analyze_with_llm(self, text: str) -> AudioPlan:
        """Uses LLM to assign a single background track and paragraph transitions."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found. Fallback to basic rule-based assignment.")
            return self._fallback_analysis(text)
            
        try:
            client = genai.Client(api_key=api_key)
            prompt = f"""
أنت مخرج صوتي سينمائي خبير. قمنا بتزويدك بنص، وقائمة بملفات موسيقى متاحة لتكون خلفية صوتية مستمرة.
المطلوب:
1. قراءة النص واختيار ملف موسيقى واحد فقط (overall_bgm_file) ليكون خلفية موسيقية للعمل بأكمله، بناءً على الطابع العام للنص.
2. تقسيم النص إلى فقرات (paragraphs).
3. تحديد فترة الصمت/الفاصل الزمني (بين 1 إلى 10 ثوانٍ كحد أقصى) قبل بداية كل فقرة (عدا الأولى) لتوفير انتقال سينمائي.
4. توضيح سبب اختيار الملف الموسيقي (bgm_reasoning) وسبب الفواصل (reasoning).

النص:
{text}

الملفات الموسيقية المتوفرة:
{', '.join(self.available_bgm)}

قم بإرجاع النتيجة ككائن JSON.
"""
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': AudioPlan,
                    'temperature': 0.7
                },
            )
            return AudioPlan.model_validate_json(response.text)
            
        except Exception as e:
            logger.error(f"LLM mapping failed: {e}. Using fallback.")
            return self._fallback_analysis(text)

    def _fallback_analysis(self, text: str) -> AudioPlan:
        """Rule-based fallback if LLM mapping is unavailable."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        bgm_file = random.choice(self.available_bgm) if self.available_bgm else "default.mp3"
        plan = AudioPlan(overall_bgm_file=bgm_file, bgm_reasoning="Fallback assignment", paragraphs=[])
        
        for p in paragraphs:
            silence = random.uniform(2.0, 5.0)
            plan.paragraphs.append(ParagraphPlan(
                text=p, 
                silence_before_sec=silence,
                reasoning="Fallback assignment"
            ))
        return plan

    def _generate_synthetic_transition(self, duration_ms: int) -> AudioSegment:
        """Generates a soft white noise swoosh as a fallback cinematic transition."""
        if duration_ms <= 0:
            return AudioSegment.silent(duration=0)
        noise = WhiteNoise().to_audio_segment(duration=duration_ms, volume=-15.0)
        return noise.fade_in(duration_ms // 2).fade_out(duration_ms // 2)

    def process_script(self, text: str, output_file: str):
        """Main pipeline to process the script into a final cinematic audio."""
        logger.info("Analyzing text content...")
        plan = self._analyze_with_llm(text)
        
        logger.info(f"Selected Background Music: {plan.overall_bgm_file} ({plan.bgm_reasoning})")
        
        voice_track = AudioSegment.silent(duration=1000) # Start with 1 sec silence
        
        temp_dir = Path("temp_cinematic")
        temp_dir.mkdir(exist_ok=True)
        
        for i, p in enumerate(plan.paragraphs):
            logger.info(f"Processing paragraph {i+1}/{len(plan.paragraphs)}")
            logger.info(f"Silence before: {p.silence_before_sec}s, Reason: {p.reasoning}")
            
            silence_ms = int(min(10.0, max(0.5, p.silence_before_sec)) * 1000)
            
            # cinematic transition effect in the background of the silence
            if i > 0:
                transition_audio = self._generate_synthetic_transition(silence_ms)
                silence_pad = AudioSegment.silent(duration=silence_ms)
                voice_track += silence_pad.overlay(transition_audio)
            else:
                voice_track += AudioSegment.silent(duration=silence_ms)
            
            # Generate Audio via TTS using the DEFAULT Narrator
            temp_wav = temp_dir / f"para_{i}.wav"
            # Hardcoded default narrator for speech
            ref_path = "/Users/alyasi/apva/data/ref/salem_podcast_clean.wav"
            
            try:
                self.tts_manager.generate_audio(
                    text=p.text,
                    output_path=str(temp_wav),
                    ref_wav=ref_path,
                    language="ar"
                )
                tts_audio = AudioSegment.from_wav(str(temp_wav))
                voice_track += tts_audio
                
            except Exception as e:
                logger.error(f"Failed to generate audio for paragraph {i}: {e}")
                
        # Now mix the voice track with the continuous background music
        bgm_path = self.bgm_dir / plan.overall_bgm_file
        if bgm_path.exists():
            bgm_audio = AudioSegment.from_file(str(bgm_path))
            
            # Reduce BGM volume to 50% max amplitude (-6dB) -> this implements the gradual rise to 50%
            bgm_audio = bgm_audio - 6.0 
            
            # Loop BGM if it's shorter than the voice track
            if len(bgm_audio) < len(voice_track):
                loop_count = math.ceil(len(voice_track) / len(bgm_audio))
                bgm_audio = bgm_audio * loop_count
                
            # Truncate to exact length of voice track + a short tail
            bgm_audio = bgm_audio[:len(voice_track) + 3000]
            
            # Fade in BGM at the start over 3 seconds for a smooth cinematic entry
            bgm_audio = bgm_audio.fade_in(3000).fade_out(3000)
            
            # Overlay voice on top of BGM
            final_mix = bgm_audio.overlay(voice_track)
        else:
            logger.warning(f"Background music {bgm_path} not found. Exporting without BGM.")
            final_mix = voice_track

        # Export final
        logger.info(f"Exporting cinematic mix to {output_file}...")
        final_mix.export(output_file, format="mp3", bitrate="192k")
        logger.info("Done!")

if __name__ == "__main__":
    sample_text = """في أعماق الفضاء السحيق، لا يُسمع سوى صدى النجوم المتلاشية. هنا تبدأ رحلتنا.

ولكن فجأة، يظهر بصيص أمل في الأفق، كوكبة جديدة تتشكل في سماء الليل لتروي قصة أخرى.

ومع ذلك، التحديات لم تنته بعد، فالرحلة نحو المجهول محفوفة بالمخاطر والأسرار التي لم تُكتشف."""
    
    base = Path(__file__).resolve().parents[2]
    processor = CinematicProcessor(bgm_dir=str(base / "assets" / "bgm"))
    processor.process_script(sample_text, "final_cinematic_output.mp3")
