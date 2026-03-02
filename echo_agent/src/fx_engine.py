import subprocess
import os
from pathlib import Path
from typing import List, Dict, Any

class FXEngine:
    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path

    def apply_effects(self, input_path: str, output_path: str, effect_cfg: Dict[str, Any]):
        """
        Applies Echo and Reverb. (Currently disabled per user request to remove echo)
        """
        import shutil
        shutil.copy(input_path, output_path)

    def concatenate_and_mix(self, segments_fx: List[str], pauses: List[float], output_path: str):
        """
        Concatenates segments with silence/pauses.
        """
        # Create a temp file for ffmpeg concat
        concat_file = Path("concat_list.txt")
        with open(concat_file, 'w', encoding='utf-8') as f:
            for i, seg_path in enumerate(segments_fx):
                f.write(f"file '{seg_path}'\n")
                if i < len(pauses):
                    # We need a silence file. We can generate it on the fly or use 'adelay'
                    # Better to generate it.
                    silence_path = f"work/silence_{pauses[i]}.wav"
                    self.generate_silence(pauses[i], silence_path)
                    f.write(f"file '{os.path.abspath(silence_path)}'\n")
                    
        cmd = [
            self.ffmpeg_path, "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-c", "pcm_s16le",
            output_path
        ]
        
        subprocess.run(cmd, check=True)
        # Cleanup
        if concat_file.exists():
            concat_file.unlink()

    def generate_silence(self, duration: float, output_path: str):
        if os.path.exists(output_path):
            return
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cmd = [
            self.ffmpeg_path, "-y",
            "-f", "lavfi", "-i", f"anullsrc=r=24000:cl=mono",
            "-t", str(duration),
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    def add_background_music(self, speech_path: str, bgm_path: str, output_path: str, bgm_volume: float = 0.35):
        """
        Mixes speech with a background music track. Matches duration to speech.
        Applies audio ducking (sidechain compression) so music lowers when someone speaks.
        """
        if not os.path.exists(bgm_path):
            print(f"BGM {bgm_path} not found. Skipping music mix.")
            import shutil
            shutil.copy(speech_path, output_path)
            return

        # [0:a] is the speech audio stream
        # [1:a] is the background music stream
        # threshold=0.015 (high sensitivity to speech), ratio=10.0 (heavy ducking)
        # attack=100ms (smooth fade down), release=2000ms (slow fade up during silence)
        filter_complex = (
            f"[0:a]volume=1[a0];"
            f"[1:a]volume={bgm_volume}[a1];"
            f"[a1][a0]sidechaincompress=threshold=0.02:ratio=8:attack=20:release=300:makeup=1[ducked];"
            f"[ducked][a0]amix=inputs=2:weights=1 1:normalize=0:duration=first"
        )

        cmd = [
            self.ffmpeg_path, "-y",
            "-i", speech_path,
            "-stream_loop", "-1", "-i", bgm_path,
            "-filter_complex", filter_complex,
            output_path
        ]
        try:
            process = subprocess.run(cmd, check=True, capture_output=True)
            print("Successfully added background music with ducking.")
        except subprocess.CalledProcessError as e:
            print(f"Error adding BGM. Stdout: {e.stdout.decode()} \n Stderr: {e.stderr.decode()}")
            import shutil
            shutil.copy(speech_path, output_path)

    def convert_to_mp3(self, input_path: str, output_path: str):
        cmd = [
            self.ffmpeg_path, "-y",
            "-i", input_path,
            "-codec:a", "libmp3lame", "-qscale:a", "2",
            output_path
        ]
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    fx = FXEngine()
    print("FX Engine ready.")
