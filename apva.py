#!/usr/bin/env python3
import sys
from pathlib import Path

# Add src to python path so modules can be found easily
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Import the main CLI from speaker module
from speaker.tts_cli import main

if __name__ == "__main__":
    import sys
    # Fix for custom argument
    if len(sys.argv) >= 4 and sys.argv[1] == "--learn":
        from speaker.tts_cli import main
        main()
    else:
        main()
