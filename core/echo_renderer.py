import os
import subprocess
from pathlib import Path
from typing import List, Optional
from core.logger import setup_logger

logger = setup_logger("EchoRenderer")

class AudioRenderer:
    """
    Handles all post-generation audio manipulation:
    - Joining WAVs
    - Silence injection/trimming
    - Background music mixing with ducking
    - MP3 Conversion
    """
    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path

    def _run_ffmpeg(self, cmd: List[str]) -> bool:
        try:
            logger.debug(f"Running FFmpeg: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, text=True, stdin=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg Error: {e.stderr}\nStdout: {e.stdout}")
            return False

    def check_ffmpeg(self) -> bool:
        import shutil
        return shutil.which(self.ffmpeg_path) is not None

    def generate_silence(self, duration_sec: float, output_path: Path):
        if output_path.exists():
            return
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.ffmpeg_path, "-y",
            "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
            "-t", str(duration_sec),
            str(output_path)
        ]
        self._run_ffmpeg(cmd)

    def concatenate_audio(self, audio_paths: List[Path], pauses: List[float], output_path: Path, work_dir: Path, intro_pause: float = 0.0):
        """
        Takes a list of audio files and a list of pauses (in seconds) to insert between them.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        concat_file = work_dir / "concat_list.txt"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(concat_file, 'w', encoding='utf-8') as f:
                if intro_pause > 0:
                    silence_path = work_dir / f"intro_silence_{intro_pause}.wav"
                    self.generate_silence(intro_pause, silence_path)
                    f.write(f"file '{silence_path.resolve()}'\n")

                for i, seg_path in enumerate(audio_paths):
                    f.write(f"file '{seg_path.resolve()}'\n")
                    # If there's a pause after this segment
                    if i < len(pauses) and pauses[i] > 0:
                        silence_path = work_dir / f"silence_{pauses[i]}.wav"
                        self.generate_silence(pauses[i], silence_path)
                        f.write(f"file '{silence_path.resolve()}'\n")

            cmd = [
                self.ffmpeg_path, "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_file),
                "-c", "pcm_s16le",
                str(output_path)
            ]
            
            if self._run_ffmpeg(cmd):
                logger.info(f"Successfully concatenated audio to {output_path}")
            else:
                logger.error("Failed to concatenate audio.")
                
        finally:
            if concat_file.exists():
                concat_file.unlink()

    def add_background_music(self, speech_path: Path, bgm_path: Path, output_path: Path, bgm_volume: float = 0.35, duck_ratio: float = 2.0):
        """
        Mixes speech with a background music track using sidechain compression (ducking).
        """
        if not bgm_path.exists():
            logger.warning(f"Background music not found at {bgm_path}. Skipping BGM mix.")
            import shutil
            shutil.copy(speech_path, output_path)
            return

        # Ducking parameters:
        # threshold=0.02 (sensitive to speech)
        # ratio=duck_ratio (how much to squash the music volume)
        # attack=20ms (how fast music drops)
        # release=300ms (how fast music recovers)
        
        filter_complex = (
            f"[0:a]volume=1[a0];"
            f"[1:a]volume={bgm_volume}[a1];"
            f"[a1][a0]sidechaincompress=threshold=0.02:ratio={duck_ratio}:attack=20:release=300:makeup=1[ducked];"
            f"[ducked][a0]amix=inputs=2:weights=1 1:normalize=0:duration=first"
        )

        cmd = [
            self.ffmpeg_path, "-y",
            "-i", str(speech_path),
            "-stream_loop", "-1", "-i", str(bgm_path),
            "-filter_complex", filter_complex,
            str(output_path)
        ]
        
        if self._run_ffmpeg(cmd):
            logger.info(f"Successfully mixed BGM with ducking to {output_path}")
        else:
            logger.error("Failed to mix BGM. Falling back to speech only.")
            import shutil
            shutil.copy(speech_path, output_path)

    def convert_to_mp3(self, input_wav: Path, output_mp3: Path):
        """Converts WAV to 192kbps MP3 for web delivery."""
        cmd = [
            self.ffmpeg_path, "-y",
            "-i", str(input_wav),
            "-codec:a", "libmp3lame", "-qscale:a", "2",
            str(output_mp3)
        ]
        if self._run_ffmpeg(cmd):
            logger.info(f"Converted {input_wav.name} to MP3.")
        else:
            logger.error(f"Failed to convert {input_wav.name} to MP3.")
