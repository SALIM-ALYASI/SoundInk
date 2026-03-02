import json
import os
from pathlib import Path
from typing import List, Dict, Any

class Director:
    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        self.presets = self._load_json(self.config_dir / "presets.json")
        
    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def generate_initial_map(self, segments: List[Dict[str, Any]], preset_name: str = "rasis_echo") -> List[Dict[str, Any]]:
        """
        Takes segments and assigns effects/pauses based on presets and segment type.
        """
        preset = self.presets.get(preset_name, {})
        voice_configs = preset.get("voices", {})
        
        director_map = []
        
        for i, seg in enumerate(segments):
            speaker = seg['speaker']
            s_type = seg['type']
            text = seg['text']
            
            v_cfg = voice_configs.get(speaker, voice_configs.get("voice1", {}))
            
            # Default decisions
            pause_after = 1.0 # Default
            echo_mode = "none"
            reverb_mode = v_cfg.get("reverb_mode", "none")
            
            # Type-based overrides
            # Type-based overrides
            if s_type == "QUESTION":
                pause_after = 2.0
            elif s_type == "PUNCHLINE":
                pause_after = 2.5
                echo_mode = "none" # Removed echo to make it calm
            elif s_type == "STATEMENT":
                pause_after = 1.0
            
            # Special case for speaker change
            if i < len(segments) - 1:
                if segments[i+1]['speaker'] != speaker:
                    pause_after += 0.5 # Extra time for turn-taking
            
            director_map.append({
                "index": i,
                "speaker": speaker,
                "text": text,
                "type": s_type,
                "pause_after_sec": pause_after,
                "echo_mode": echo_mode,
                "reverb_mode": reverb_mode,
                "music_ducking": 0.2
            })
            
        return director_map

    def save_map(self, d_map: List[Dict[str, Any]], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(d_map, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    director = Director("../config")
    test_segs = [
        {"speaker": "voice1", "text": "أهلاً بك في عالم الذكاء الاصطناعي.", "type": "STATEMENT"},
        {"speaker": "voice2", "text": "هل تعتقد أنه سيعوضنا يوماً ما؟", "type": "QUESTION"},
        {"speaker": "voice1", "text": "الأثر... باقٍ.", "type": "PUNCHLINE"}
    ]
    res = director.generate_initial_map(test_segs)
    print(json.dumps(res, ensure_ascii=False, indent=2))
