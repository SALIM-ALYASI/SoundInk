#!/bin/bash

source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate ai-agent

uvicorn app:app --reload --port 5000



# ./run_env.sh


# python3 src/brain/project_analyzer.py