#!/bin/bash

cd "$(dirname "$0")" || exit 1

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

pip install -q -r requirements.txt

export COQUI_TOS_AGREED=1

uvicorn app:app --reload --host 0.0.0.0 --port 5050




# du -sh Audio


# python3 src/brain/project_analyzer.py 