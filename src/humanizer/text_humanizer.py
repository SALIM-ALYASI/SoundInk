import json
import random
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = BASE_DIR / "configs"

def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Empty config file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

PERSONA = load_json(CONFIG_DIR / "persona.json")
PAUSE = load_json(CONFIG_DIR / "pause_rules.json")

STARTERS = [
    "شوف",
    "طيب خلّنا نتكلم عن",
    "خلّني أوضح",
    "ببساطة",
]

# 🔹 تنظيف النص قبل أي معالجة
def clean_text(text: str) -> str:
    # حذف ماركرات الحماية [[[PROT_000]]]
    text = re.sub(r"\[\[\[.*?\]\]\]", "", text)

    # حذف ellipsis
    text = text.replace("…", "")

    # حذف النقاط تمامًا
    text = re.sub(r"[.]", "", text)

    # إزالة علامات استفهام وتعجب (لو ما تبغى تقسيم)
    text = text.replace("؟", "")
    text = text.replace("!", "")

    # إزالة المسافات المكررة
    text = re.sub(r"\s+", " ", text).strip()

    return text


# 🔹 تقسيم جمل أكثر استقرار (اعتمادًا على pause_rules بدل النقاط)
def split_sentences(text):
    text = clean_text(text)
    sentences = re.split(r"\n+", text)  # تقسيم فقط على سطر جديد
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def add_starter(sentence, probability=0.8):
    if random.random() < probability:
        return random.choice(STARTERS) + " " + sentence
    return sentence


def add_pause(sentence):
    # تحويل الفاصلة العربية لوقفة صوتية
    sentence = sentence.replace("،", PAUSE["short_pause"])
    sentence = sentence.replace(",", PAUSE["short_pause"])
    return sentence


def humanize(text):
    sentences = split_sentences(text)
    output = []
    last_starter = None

    for i, s in enumerate(sentences):

        # starter في أول جملة غالبًا
        if i == 0:
            s = add_starter(s, probability=0.9)
            last_starter = s.split()[0]
        else:
            if random.random() < 0.4:
                candidate = random.choice(STARTERS)
                if candidate != last_starter:
                    s = candidate + " " + s

        s = add_pause(s)
        output.append(s)

    # نستخدم medium_pause بدل النقاط نهائيًا
    return PAUSE["medium_pause"].join(output)


if __name__ == "__main__":
    sample = "الذكاء الاصطناعي غير طريقة صناعة المحتوى بالكامل. هذا التطور سريع جدا."
    print(humanize(sample))