#!/bin/bash
echo "Starting APVA Audio Generation..."

# 1. Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate apva310

# 2. Run the TTS generation script with required arguments
python apva.py --text-file input.txt --out final_audio.wav --split --concat --lexicon

# 3. Notify completion
echo ""
echo "✅ Done! The final generated audio has been saved to: outputs/raw/final_audio_full.wav"
