import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.cinematic.processor import CinematicProcessor

def main():
    parser = argparse.ArgumentParser(description="Generate Cinematic Audio from Text")
    parser.add_argument("--script", type=str, required=True, help="Path to text script file (e.g., script.txt)")
    parser.add_argument("--voice_bank", type=str, default=str(Path(__file__).parent / "assets" / "bgm"), help="Directory containing voice MP3 files")
    parser.add_argument("--transitions", type=str, default="", help="Directory containing transition SFX (optional)")
    parser.add_argument("--output", type=str, default="cinematic_output.mp3", help="Output final audio file path")

    args = parser.parse_args()

    # Load text
    script_path = Path(args.script)
    if not script_path.exists():
        print(f"Error: Script file not found: {args.script}")
        return

    text = script_path.read_text(encoding='utf-8')

    # Run processing
    print(f"Starting cinematic generation for script {args.script}...")
    processor = CinematicProcessor(
        bgm_dir=args.voice_bank,
        transitions_dir=args.transitions if args.transitions else None
    )

    processor.process_script(text, args.output)
    print(f"\n[+] Success! Cinematic final audio saved to {args.output}")

if __name__ == "__main__":
    main()
