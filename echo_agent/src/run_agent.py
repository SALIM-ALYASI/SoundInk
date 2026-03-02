import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analyzer import ScriptAnalyzer
from director import Director
from engine_tts import TTSEngine
from fx_engine import FXEngine

# Configuration
BASE_DIR = Path("/Users/alyasi/apva/echo_agent")
PYTHON_BIN = sys.executable
CLI_PATH = "/Users/alyasi/apva/src/speaker/tts_cli.py"

def run_pipeline(script_path: str, output_name: str = "final_output", tts_manager=None, base_dir=None):
    if base_dir:
        global BASE_DIR
        BASE_DIR = Path(base_dir)
        
    print("--- Echo Director AI V0 Pipeline Started ---")
    
    # 1. Analyze
    print("[1/5] Analyzing Script...")
    analyzer = ScriptAnalyzer(str(BASE_DIR / "config/lexicon.txt"))
    with open(script_path, 'r', encoding='utf-8') as f:
        script_content = f.read()
    segments = analyzer.parse_script(script_content)
    
    # 2. Map
    print("[2/5] Generating Director Map...")
    director = Director(str(BASE_DIR / "config"))
    d_map = director.generate_initial_map(segments)
    
    reports_dir = BASE_DIR / "output" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    director.save_map(d_map, str(reports_dir / f"{output_name}_map.json"))
    
    # 3. TTS
    print("[3/5] Generating Raw Audio (TTS)...")
    engine = TTSEngine(PYTHON_BIN, CLI_PATH, manager=tts_manager)
    raw_audio_dir = BASE_DIR / "work" / output_name / "tts_raw"
    raw_audio_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = engine.generate_raw_audio(segments, str(raw_audio_dir))
    
    # 4. FX
    print("[4/5] Applying Effects (FX)...")
    fx = FXEngine()
    mixed_dir = BASE_DIR / "work" / output_name / "mix"
    os.makedirs(mixed_dir, exist_ok=True)
    
    fx_paths = []
    pauses = []
    for i, seg_map in enumerate(d_map):
        raw_p = raw_paths[i]
        fx_p = str(mixed_dir / f"seg_{i:04d}_fx.wav")
        fx.apply_effects(raw_p, fx_p, seg_map)
        fx_paths.append(fx_p)
        pauses.append(seg_map['pause_after_sec'])
        
    # 5. Mix
    print("[5/5] Final Mixing with Background Music...")
    final_wav_no_bgm = str(mixed_dir / f"{output_name}_no_bgm.wav")
    
    # ensure output/final exists
    final_out_dir = BASE_DIR / "output" / "final"
    final_out_dir.mkdir(parents=True, exist_ok=True)
    
    final_wav = str(final_out_dir / f"{output_name}.wav")
    final_mp3 = str(final_out_dir / f"{output_name}.mp3")
    
    # 5.1 Concatenate speech and pauses
    fx.concatenate_and_mix(fx_paths, pauses, final_wav_no_bgm)
    
    # 5.2 Add Background Music
    bgm_path = str(BASE_DIR / "assets/music/calm_music.mp3")
    fx.add_background_music(final_wav_no_bgm, bgm_path, final_wav, bgm_volume=0.08)
    
    # 5.3 Convert to MP3
    fx.convert_to_mp3(final_wav, final_mp3)
    
    print(f"--- Pipeline Finished Successfully! ---")
    print(f"Final MP3: {final_mp3}")
    print(f"Report: {BASE_DIR}/output/reports/{output_name}_map.json")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        script = sys.argv[1]
        out_name = Path(script).stem
    else:
        script = str(BASE_DIR / "input/scripts/test_v0.txt")
        out_name = "final_output"
        
    run_pipeline(script, out_name)
