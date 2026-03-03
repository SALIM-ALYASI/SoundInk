#!/bin/bash
echo "Starting Celery worker for SoundInk..."

# Activate conda environment
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh 2>/dev/null || source $(conda info --base)/etc/profile.d/conda.sh
conda activate apva310

# Create logs dir
mkdir -p logs

# Start Celery worker in background 
nohup celery -A core.celery_app worker --loglevel=info > logs/celery.log 2>&1 &
PID=$!

echo "Celery worker started successfully in the background!"
echo "PID: $PID"
echo "Logs are being written to: logs/celery.log"
echo "You can view logs with: tail -f logs/celery.log"
