import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
print("Torch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())

try:
    from TTS.api import TTS
    print("Loading TTS model...")
    MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts = TTS(MODEL_NAME)
    print("TTS Loaded Successfully!")
except Exception as e:
    print("Error:", e)
