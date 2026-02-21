from pathlib import Path
from TTS.api import TTS

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "outputs" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REF_WAV = BASE_DIR / "data" / "ref" / "salem_master.wav"

if not REF_WAV.exists():
    raise FileNotFoundError(f"Reference file not found: {REF_WAV}")

tts = TTS(MODEL_NAME)

def tts_generate(text: str, out_name: str = "test.wav"):
    out_path = OUT_DIR / out_name
    tts.tts_to_file(
        text=text,
        file_path=str(out_path),
        speaker_wav=str(REF_WAV),
        language="ar"
    )
    return out_path

if __name__ == "__main__":
    sample = """اللهم اجعل أول أيام رمضان بداية خير لنا،
واكتب لنا فيه الرحمة والمغفرة والعتق من النار،
ولا تجعلنا من المحرومين من فضلك"""
    path = tts_generate(sample, "salem_test.wav")
    print("Saved:", path)