import re
import random

# كلمات/تراكيب تخفف الرسمية (سعودي خفيف)
PHRASE_MAP = [
    ("لكن", "بس"),
    ("جدا", "مرّة"),
    ("كذلك", "برضه"),
    ("أيضا", "بعد"),
    ("لذلك", "عشان كذا"),
    ("هذا الأمر", "هالموضوع"),
    ("هذه الفكرة", "هالفكرة"),
    ("هذا الشيء", "هالشي"),
    ("يجب", "لازم"),
    ("ينبغي", "المفروض"),
    ("سوف", "بـ"),
]

# بادئات/جمل حوارية خفيفة (نستخدمها باحتمال بسيط)
STARTERS = [
    "شوف…",
    "طيب…",
    "خلّني أوضح…",
    "ببساطة…",
    "بصراحة…",
]

def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _soften_phrases(text: str) -> str:
    # استبدالات بسيطة بدون ما نخرب المعنى
    for a, b in PHRASE_MAP:
        text = text.replace(a, b)
    return text

def _break_long_sentences(text: str) -> str:
    """
    يكسر الجمل الطويلة بشكل طبيعي:
    - عند (،) و (و) إذا الجملة طويلة
    - يحول النقطة إلى وقفة أنعم (…)
    """
    # وحّد النقاط
    text = text.replace("...", "…")
    text = re.sub(r"\.\s*", "… ", text)  # النقطة = قفلة حادة، نخليها أنعم
    text = re.sub(r"\s*…\s*", "… ", text).strip()

    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            lines.append("")
            continue

        # إذا السطر طويل، نكسّره عند الفواصل
        if len(line) > 120:
            # كسر عند الفاصلة العربية
            line = line.replace("،", "…\n")
            # كسر عند " و" (بحذر)
            line = re.sub(r"\sو(?=\S)", "\nو", line)
        lines.append(line)

    return "\n".join(lines)

def _add_light_starters(text: str, probability: float = 0.25) -> str:
    """
    يضيف 'شوف/طيب...' للجمل الأولى أو بعد فراغات
    باحتمال بسيط حتى ما يصير مزعج.
    """
    chunks = [c.strip() for c in text.split("\n")]

    out = []
    for i, c in enumerate(chunks):
        if not c:
            out.append("")
            continue

        # لا نضيف ستارتر للدعاء أو النصوص الدينية غالبًا
        is_dua = any(w in c for w in ["اللهم", "يا رب", "رب", "استغفر", "سبحان"])
        if is_dua:
            out.append(c)
            continue

        if (i == 0 or (i > 0 and chunks[i-1] == "")) and random.random() < probability:
            out.append(f"{random.choice(STARTERS)} {c}")
        else:
            out.append(c)

    return "\n".join(out)

def _final_tuning(text: str) -> str:
    # رتّب المسافات حول الفواصل
    text = re.sub(r"\s*,\s*", "، ", text)
    text = re.sub(r"\s*،\s*", "، ", text)
    # قفلة حوارية ناعمة: لا تخلي السطر ينتهي بنقطة حادة
    text = re.sub(r"[\.!؟]\s*$", "…", text)
    text = re.sub(r"\s+", " ", text)
    # رجّع أسطر جديدة بوضوح (بعد ما سوينا تسوية مسافات)
    text = text.replace(" \n", "\n").replace("\n ", "\n")
    return text.strip()

def smart_filter_chatty(text: str, starters: bool = True) -> str:
    text = _normalize(text)
    text = _soften_phrases(text)
    text = _break_long_sentences(text)
    if starters:
        text = _add_light_starters(text, probability=0.25)
    text = _final_tuning(text)
    return text