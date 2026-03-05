import sys
import os
import time
from core.pipeline import EchoPipeline

# Ensure models can load directly in local testing
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

def test_pipeline():
    try:
        from src.speaker.inference_manager import tts_manager
    except ImportError as e:
        print(f"Failed to load TTS Manager: {e}")
        sys.exit(1)

    pipeline = EchoPipeline(tts_manager=tts_manager)
    
    text = "مرحبا بك في عالم المعرفة. هذا اختبار لسرعة ونبرة الصوت الجديدة. نتمنى أن ينال إعجابكم."
    
    print("--- Testing NORMAL Mode ---")
    try:
        res_normal = pipeline.generate_dual_output(text, mode="normal")
        print("Normal mode output:", res_normal)
    except Exception as e:
        print("Normal Error:", e)
        
    print("\n--- Testing CINEMATIC Mode ---")
    try:
        res_cinematic = pipeline.generate_dual_output(text, mode="cinematic")
        print("Cinematic mode output:", res_cinematic)
    except Exception as e:
        print("Cinematic Error:", e)
    
if __name__ == "__main__":
    test_pipeline()
