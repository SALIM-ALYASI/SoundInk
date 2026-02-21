import random
import re

EM_OPEN = "<em>"
EM_CLOSE = "</em>"
BREATH = "<breath>"

def add_emphasis(sentence: str) -> str:
    words = sentence.split()
    if len(words) < 5:
        return sentence

    # اختر كلمة وسطية (لا أول/لا آخر) لتجنب شكل غريب
    idx = random.randint(1, len(words) - 2)
    words[idx] = f"{EM_OPEN}{words[idx]}{EM_CLOSE}"
    return " ".join(words)

def add_breath_markers(text: str, probability: float = 0.6) -> str:
    # نحط breath بعد الوقفة المتوسطة أحيانًا
    parts = [p.strip() for p in text.split("...") if p.strip()]
    out = []
    for i, p in enumerate(parts):
        out.append(p)
        if i < len(parts) - 1 and random.random() < probability:
            out.append(BREATH)
    return " ... ".join(out)

def performance_enhance(text: str, emphasis_prob: float = 0.6) -> str:
    parts = [p.strip() for p in text.split("...") if p.strip()]
    enhanced = []

    for p in parts:
        s = p
        if random.random() < emphasis_prob:
            s = add_emphasis(s)
        enhanced.append(s)

    result = " ... ".join(enhanced)
    result = add_breath_markers(result, probability=0.6)
    return result

def strip_markers_for_tts(text: str) -> str:
    """إذا محرك الصوت ما يدعم العلامات، نشيلها ونحوّل breath لوقفة نظيفة بدون تكرار."""
    text = text.replace(BREATH, " ... ")   # نخليه نفس الوقفة المتوسطة بدل ما يصير .. جنب ...
    text = text.replace(EM_OPEN, "")
    text = text.replace(EM_CLOSE, "")

    # تنظيف تكرار الوقفات: "... ...", ".. ..", "... .." إلخ
    text = re.sub(r"(\.\s*){6,}", "... ", text)   # أي تكرار كثير للنقاط
    text = re.sub(r"\.\.\.\s*\.\.\.", "... ", text)
    text = re.sub(r"\.\.\s*\.\.\.", "... ", text)
    text = re.sub(r"\.\.\.\s*\.\.", "... ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text