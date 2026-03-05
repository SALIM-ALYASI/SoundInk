import os
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

print("Loading torch...")
import torch
print(f"Torch loaded: {torch.__version__}")

try:
    print("Importing TTS...")
    from TTS.api import TTS
    print("TTS module imported.")
except ImportError as e:
    print(f"Failed to import TTS: {e}")
    sys.exit(1)

print("Initializing XTTS v2...")
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
try:
    tts = TTS(MODEL_NAME).to("cpu")
    print("TTS Loaded Successfully to CPU!")
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit(1)
