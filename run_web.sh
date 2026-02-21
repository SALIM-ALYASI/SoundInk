#!/bin/bash
echo "Starting SoundInk Web Interface..."

# 1. Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate apva310

# 2. Start the Flask server
echo "Server is starting. Open your browser and go to: http://localhost:5000"
echo "Press Ctrl+C to stop the server."
python app.py
