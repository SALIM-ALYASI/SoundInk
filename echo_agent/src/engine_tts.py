import os
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any

class TTSEngine:
    def __init__(self, python_path: str, tts_cli_path: str, manager=None):
        self.python_path = python_path
        self.tts_cli_path = tts_cli_path
        self.manager = manager
        self.default_refs = {
            "voice1": "/Users/alyasi/apva/data/ref/salem_podcast_clean.wav",
            "voice2": "/Users/alyasi/apva/data/ref/ref_master.wav" # Fallback or needs to be provided
        }

    def generate_raw_audio(self, segments: List[Dict[str, Any]], output_dir: str) -> List[str]:
        """
        Generates raw WAV files for each segment using the tts_cli.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        
        for i, seg in enumerate(segments):
            speaker = seg['speaker']
            text = seg['text']
            
            ref_path = self.default_refs.get(speaker, self.default_refs["voice1"])
            out_name = f"seg_{i:04d}_{speaker}.wav"
            out_full_path = os.path.join(output_dir, out_name)
            
            # Construct command
            # Using the existing tts_cli.py because it already handles XTTS and cleaning
            cmd = [
                self.python_path,
                self.tts_cli_path,
                "--text", text,
                "--ref", ref_path,
                "--out", out_name,
                "--no-clean" # We might want to clean, but analyzer might have split things already
            ]
            
            print(f"Generating audio for segment {i} ({speaker})...")
            
            if self.manager:
                try:
                    self.manager.generate_audio(
                        text=text,
                        output_path=out_full_path,
                        ref_wav=ref_path
                    )
                    paths.append(out_full_path)
                except Exception as e:
                    print(f"Error generating audio for segment {i} in RAM: {e}")
                    raise Exception(f"TTS process failed for segment {i}")
            else:
                try:
                    env = os.environ.copy()
                    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
                    
                    process = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
                    
                    # tts_cli.py saves in BASE_DIR / "storage" / "outputs" / "raw"
                    original_out_path = os.path.join("/Users/alyasi/apva/storage/outputs/raw", out_name)
                    
                    if os.path.exists(original_out_path):
                        import shutil
                        shutil.move(original_out_path, out_full_path)
                        paths.append(out_full_path)
                    else:
                        print(f"File not found after TTS generation: {original_out_path}")
                        print(f"TTS STDOUT: {process.stdout}")
                        print(f"TTS STDERR: {process.stderr}")
                        raise Exception(f"Failed to find output for segment {i}")
                except subprocess.CalledProcessError as e:
                    print(f"Error generating audio for segment {i}: {e.stderr}")
                    print(f"Stdout: {e.stdout}")
                    raise Exception(f"TTS process failed for segment {i}")
                
        return paths

if __name__ == "__main__":
    import sys
    # Example usage
    PYTHON_BIN = sys.executable
    CLI_PATH = "/Users/alyasi/apva/src/speaker/tts_cli.py"
    
    engine = TTSEngine(PYTHON_BIN, CLI_PATH)
    test_segments = [
        {"speaker": "voice1", "text": "أهلاً بك في عالم الذكاء الاصطناعي.", "type": "STATEMENT"},
        {"speaker": "voice2", "text": "شكراً جزيلاً، أنا متحمس جداً.", "type": "STATEMENT"}
    ]
    
    # paths = engine.generate_raw_audio(test_segments, "/Users/alyasi/apva/storage/work/tts_raw")
    print("Engine ready.")
