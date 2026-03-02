import re
import json
import os
from typing import List, Dict, Any

class ScriptAnalyzer:
    def __init__(self, lexicon_path: str):
        self.lexicon = self._load_lexicon(lexicon_path)
    
    def _load_lexicon(self, path: str) -> List[str]:
        if not os.path.exists(path):
            return []
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def classify_sentence(self, text: str) -> str:
        """Classify a sentence into QUESTION, PUNCHLINE, or STATEMENT."""
        text = text.strip()
        if not text:
            return "STATEMENT"
        
        # Check if question
        if '؟' in text or '?' in text or any(word in text.split() for word in ['هل', 'كيف', 'لماذا', 'متى', 'أين', 'من', 'ماذا']):
            return "QUESTION"
            
        # Check for punchlines
        has_punchline_word = any(word in text for word in self.lexicon)
        words_count = len(text.split())
        
        if has_punchline_word or words_count <= 4: # Short powerful sentences
            return "PUNCHLINE"
            
        return "STATEMENT"

    def parse_script(self, script_text: str) -> List[Dict[str, Any]]:
        """
        Parses the script supporting tags like [Voice1]: Hello
        Returns a list of parsed segments.
        """
        lines = script_text.strip().split('\n')
        segments = []
        current_speaker = "voice1" # Default assumed
        speaker_pattern = re.compile(r'^\[(.*?)\]\s*:\s*(.*)$')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            match = speaker_pattern.match(line)
            if match:
                raw_speaker = match.group(1).strip().lower()
                text = match.group(2).strip()
                # Map logical names if needed
                if raw_speaker in ['صوت 1', 'صوت1', 'voice1', 'voice 1', 'narrator']:
                    current_speaker = "voice1"
                elif raw_speaker in ['صوت 2', 'صوت2', 'voice2', 'voice 2', 'guest']:
                    current_speaker = "voice2"
                else:
                    # Generic mapping or keep as is
                    current_speaker = raw_speaker.replace(' ', '')
            else:
                text = line
            
            # Split paragraph into sentences. Simple heuristic using punctuation
            sentences = re.split(r'(?<=[.!?؟])\s+', text)
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                
                s_type = self.classify_sentence(sentence)
                
                segments.append({
                    "speaker": current_speaker,
                    "text": sentence.strip(),
                    "type": s_type
                })
                
        return segments

if __name__ == "__main__":
    # Test
    analyzer = ScriptAnalyzer("../config/lexicon.txt")
    test_script = """
[صوت 1]: أهلًا بك في هذه الحلقة.
[صوت 2]: مرحبًا بكم جميعًا! اليوم سنتحدث عن موضوع مهم.
[صوت 1]: هل أنتم مستعدون؟
[صوت 2]: البداية ستكون من هنا.
    """
    res = analyzer.parse_script(test_script)
    print(json.dumps(res, ensure_ascii=False, indent=2))
