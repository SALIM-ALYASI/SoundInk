#!/usr/bin/env python3
import argparse
import json
import re
import random
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple, Union

# -----------------------------
# Safe import for TTS
# -----------------------------
try:
    from TTS.api import TTS  # type: ignore
except Exception as e:
    raise SystemExit(
        "\n❌ ما لقيت مكتبة TTS داخل البيئة الحالية.\n"
        "✅ الحل السريع:\n"
        "  1) فعّل بيئة الكوندا اللي فيها TTS (مثلاً: conda activate apva310)\n"
        "  2) ثم شغّل نفس الأمر.\n\n"
        "لو تبي تتأكد:\n"
        "  python3 -c \"import TTS; print(TTS.__file__)\"\n"
        f"\nالخطأ الأصلي: {e}\n"
    )

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# project_root = .../apva  (لأن الملف داخل src/speaker/tts_cli.py)
BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"

OUT_DIR = BASE_DIR / "outputs" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_REF = BASE_DIR / "data" / "ref" / "salem_podcast_clean.wav"
DEFAULT_LEXICON_FILE = BASE_DIR / "configs" / "lexicon.json"
DEFAULT_SEG_CONFIG = BASE_DIR / "configs" / "segmentation.json"

# -----------------------------
# Helpers: sys.path fix (important for speaker.smart_seg imports)
# -----------------------------
def _ensure_src_on_syspath() -> None:
    s = str(SRC_DIR)
    if s not in sys.path:
        sys.path.insert(0, s)

# -----------------------------
# Smart Segmentation import (robust)
# -----------------------------
SegFunc = Callable[[str, Union[str, Path, Dict[str, Any]]], Any]

def _load_segment_text_func() -> Optional[SegFunc]:
    """
    يحاول يحمّل segment_text من:
      1) speaker.smart_seg.engine  (الطريقة الصحيحة داخل مشروعك)
      2) smart_seg.engine          (لو كنت مركبه كحزمة)
    """
    _ensure_src_on_syspath()

    try:
        from speaker.smart_seg.engine import segment_text  # type: ignore
        return segment_text  # type: ignore
    except Exception:
        pass

    try:
        from smart_seg.engine import segment_text  # type: ignore
        return segment_text  # type: ignore
    except Exception:
        return None

SEGMENT_TEXT = _load_segment_text_func()

# -----------------------------
# Optional: CAMeL Tools (Disambiguator-based diacritization)
# -----------------------------
try:
    from camel_tools.disambig.mle import MLEDisambiguator  # type: ignore
    from camel_tools.tokenizers.word import simple_word_tokenize  # type: ignore
    _DISAMBIG = MLEDisambiguator.pretrained()
except Exception:
    _DISAMBIG = None

_ONLY_PUNCT_RE = re.compile(r"^[\s\.\,\!\?\:;\u060C\u061B\u061F\u2026]+$")  # includes ، ؛ ؟ …

def camel_diacritize_light(text: str) -> str:
    """
    تشكيل خفيف عبر CAMeL:
    - يشكّل
    - ثم يشيل (التنوين + السكون) لأنهم يزيدون الطابع الروبوتي
    """
    if _DISAMBIG is None:
        return text

    toks = simple_word_tokenize(text)
    sent = _DISAMBIG.disambiguate(toks)

    out: List[str] = []
    for w in sent:
        if not w.analyses:
            out.append(w.word)
        else:
            out.append(w.analyses[0].analysis.diac)

    s = " ".join(out)
    s = s.replace("ً", "").replace("ٌ", "").replace("ٍ", "")
    s = s.replace("ْ", "")
    return s

def diacritize_selective(text: str, triggers: List[str]) -> str:
    """
    يشكّل فقط الأسطر التي تحتوي كلمات حساسة (triggers)
    """
    if _DISAMBIG is None:
        return text

    parts = re.split(r"(\n+)", text)
    out: List[str] = []
    for p in parts:
        if any(t in p for t in triggers):
            out.append(camel_diacritize_light(p))
        else:
            out.append(p)
    return "".join(out)

# -----------------------------
# Chatty smart filter
# -----------------------------
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
]

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
    for a, b in PHRASE_MAP:
        text = text.replace(a, b)
    return text

def _break_long_sentences(text: str) -> str:
    text = text.replace("...", "…")
    text = re.sub(r"\.\s*", "… ", text)
    text = re.sub(r"\s*…\s*", "… ", text).strip()

    lines: List[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            lines.append("")
            continue
        if len(line) > 120:
            line = line.replace("،", "…\n")
            line = re.sub(r"\sو(?=\S)", "\nو", line)
        lines.append(line)

    return "\n".join(lines)

def _add_light_starters(text: str, probability: float = 0.20) -> str:
    chunks = [c.strip() for c in text.split("\n")]
    out: List[str] = []
    for i, c in enumerate(chunks):
        if not c:
            out.append("")
            continue

        is_dua = any(w in c for w in ["اللهم", "يا رب", "رب", "استغفر", "سبحان"])
        if is_dua:
            out.append(c)
            continue

        if (i == 0 or (i > 0 and chunks[i - 1] == "")) and random.random() < probability:
            out.append(f"{random.choice(STARTERS)} {c}")
        else:
            out.append(c)

    return "\n".join(out)

def _final_tuning(text: str) -> str:
    text = re.sub(r"\s*,\s*", "، ", text)
    text = re.sub(r"\s*،\s*", "، ", text)
    text = re.sub(r"[\.!؟]\s*$", "…", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.replace(" \n", "\n").replace("\n ", "\n")
    return text.strip()

def smart_filter_chatty(text: str, starters: bool = True) -> str:
    text = _normalize(text)
    text = _soften_phrases(text)
    text = _break_long_sentences(text)
    if starters:
        text = _add_light_starters(text, probability=0.20)
    text = _final_tuning(text)
    return text

# -----------------------------
# Arabic clean helpers (strong)
# -----------------------------
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE,
)

try:
    from num2words import num2words
except ImportError:
    num2words = None

def digits_to_ar_words(text: str) -> str:
    """
    يبحث عن أي أرقام بالنص ويحولها لنص عربي مقروء باستخدام num2words للتحكم النحوي الممتاز
    """
    if num2words is None:
        # Fallback to naive if not installed
        AR_NUMS = {"0": "صفر", "1": "واحد", "2": "اثنين", "3": "ثلاثة", "4": "أربعة",
                   "5": "خمسة", "6": "ستة", "7": "سبعة", "8": "ثمانية", "9": "تسعة"}
        return "".join(AR_NUMS.get(ch, ch) for ch in text)

    def _replace_num(match):
        num_str = match.group(0).replace(",", "")
        try:
            return num2words(int(num_str), lang="ar")
        except ValueError:
            return match.group(0)

    # يطابق الحبوب الرقمية كاملة مثل 1200 أو 2,025
    return re.sub(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\b\d+\b", _replace_num, text)



def clean_for_ar_tts(text: str) -> str:
    t = text.strip()
    t = digits_to_ar_words(t)
    t = _EMOJI_RE.sub(" ", t)
    t = re.sub(r"[A-Za-z]+", " ", t)
    t = re.sub(r"[%@#_=+\/*\\<>[\]{}()|~^$`]", " ", t)
    t = t.replace("...", "…")
    t = re.sub(r"[؛;:]+", "،", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# -----------------------------
# Lexicon from JSON file
# -----------------------------
def load_lexicon_map(lexicon_file: Path) -> Dict[str, str]:
    if not lexicon_file.exists():
        return {}

    data = json.loads(lexicon_file.read_text(encoding="utf-8"))
    out: Dict[str, str] = {}

    if isinstance(data, dict):
        simple_pairs = {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}
        if len(simple_pairs) >= 1 and len(simple_pairs) == len(data):
            return simple_pairs

        for key in ["tribes_pronunciation", "names_pronunciation", "misc_pronunciation", "pronunciation"]:
            arr = data.get(key)
            if isinstance(arr, list):
                for item in arr:
                    if not isinstance(item, dict):
                        continue
                    orig = item.get("original")
                    fmt = item.get("formatted")
                    if isinstance(orig, str) and isinstance(fmt, str) and orig.strip() and fmt.strip():
                        out[orig.strip()] = fmt.strip()

    return out

def apply_lexicon(text: str, lex_map: Dict[str, str]) -> str:
    if not lex_map:
        return text

    # الأطول أولاً
    items = sorted(lex_map.items(), key=lambda kv: len(kv[0]), reverse=True)

    # حدود بسيطة (مسافة/بداية/نهاية/ترقيم عربي)
    boundary = r"[\s\.,،…:؛$begin:math:text$$end:math:text$$begin:math:display$$end:math:display$\{\}\"'«»]"

    def pat(word: str) -> str:
        w = re.escape(word)
        return rf"(?:(?<=^)|(?<={boundary})){w}(?:(?=$)|(?={boundary}))"

    for orig, fmt in items:
        text = re.sub(pat(orig), fmt, text)

    return text

# -----------------------------
# Unknown Words Extraction
# -----------------------------
def check_for_unknown_words(text: str, pending_file: Path):
    """
    يبحث عن أي كلمات إنجليزية تم نسيان إضافتها للقاموس،
    ويحفظها في ملف pending_words.json لسهولة الوصول لها لاحقاً.
    """
    # نبحث عن كلمات أجنبية (انجليزية)
    # نستخدم تعبير للبحث عن الكلمات التي تحتوي على أحرف إنجليزية
    unknowns = re.findall(r"\b[A-Za-z]+\b", text)
    if not unknowns:
        return

    # تنظيف وتوحيد الكلمات
    unique_unknowns = set(w for w in unknowns)
    if not unique_unknowns:
        return

    pending_data = []
    if pending_file.exists():
        try:
            pending_data = json.loads(pending_file.read_text(encoding="utf-8"))
            if not isinstance(pending_data, list):
                pending_data = []
        except:
            pass

    existing_set = set(pending_data)
    added = False
    for word in unique_unknowns:
        if word not in existing_set:
            pending_data.append(word)
            added = True

    if added:
        pending_file.parent.mkdir(parents=True, exist_ok=True)
        pending_file.write_text(json.dumps(pending_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n⚠️ تنبيه: تم العثور على كلمات أجنبية غير موجودة في القاموس. تم حفظها في:\n   {pending_file}")
        print("   لإضافتها للقاموس بسرعة، استخدم الأمر:\n   python apva.py --learn \"الكلمة\" \"النطق_العربي\"\n")

# -----------------------------
# Chunking fallback (non-smart)
# -----------------------------
def split_to_chunks(text: str, max_chars: int = 150) -> List[str]:
    parts = re.split(r"(?<=[\.\!\؟\?\n])\s+|(?<=…)\s+|(?<=،)\s+", text)
    parts = [p.strip() for p in parts if p.strip()]

    chunks: List[str] = []
    cur = ""

    def push_cur():
        nonlocal cur
        if cur.strip():
            chunks.append(cur.strip())
        cur = ""

    for p in parts:
        while len(p) > max_chars:
            cut = p.rfind(" ", 0, max_chars)
            if cut == -1:
                cut = max_chars
            piece = p[:cut].strip()
            if not piece.endswith(("…", "،")):
                piece += "…"
            chunks.append(piece)
            p = p[cut:].strip()

        if not cur:
            cur = p
        elif len(cur) + 1 + len(p) <= max_chars:
            cur = f"{cur} {p}"
        else:
            push_cur()
            cur = p

    push_cur()

    out: List[str] = []
    for c in chunks:
        c = c.strip()
        if not c or _ONLY_PUNCT_RE.match(c):
            continue
        if not c.endswith(("…", "،", ".", "!", "؟")):
            c += "…"
        out.append(c)

    return out

# -----------------------------
# Segments finalizer
# -----------------------------
def finalize_segments(segs: List[str]) -> List[str]:
    out: List[str] = []
    for s in segs:
        s = (s or "").strip()
        if not s:
            continue
        if _ONLY_PUNCT_RE.match(s):
            continue
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"[،]+\s*$", "،", s)
        s = re.sub(r"(…)+\s*$", "…", s)
        out.append(s)
    return out

# -----------------------------
# IO helpers
# -----------------------------
def load_text(text: Optional[str], text_file: Optional[str]) -> str:
    if text and text.strip():
        return text.strip()

    if text_file:
        p = Path(text_file).expanduser()
        if not p.is_absolute():
            p = (BASE_DIR / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Text file not found: {p}")
        return p.read_text(encoding="utf-8").strip()

    raise ValueError("Provide --text or --text-file")

def resolve_path(p: str) -> Path:
    ref = Path(p).expanduser()
    if not ref.is_absolute():
        ref = (BASE_DIR / ref).resolve()
    return ref

# -----------------------------
# WAV concat helpers
# -----------------------------
def _silence_frames(n_frames: int, sampwidth: int, nchannels: int) -> bytes:
    return b"\x00" * (n_frames * sampwidth * nchannels)

def concat_wavs_with_pads(in_paths: List[Path], out_path: Path, pads_ms: List[int]) -> None:
    """
    pads_ms: طولها لازم يكون len(in_paths)-1
    """
    if not in_paths:
        raise ValueError("No wav parts to concat")

    if len(in_paths) >= 2 and len(pads_ms) != len(in_paths) - 1:
        raise ValueError(f"pads_ms length mismatch: expected {len(in_paths)-1} got {len(pads_ms)}")

    params0 = None
    frames_out: List[bytes] = []

    for idx, p in enumerate(in_paths):
        with wave.open(str(p), "rb") as w:
            params = w.getparams()
            if params0 is None:
                params0 = params
            else:
                if (params.nchannels, params.sampwidth, params.framerate) != (params0.nchannels, params0.sampwidth, params0.framerate):
                    raise ValueError(
                        f"WAV params mismatch in {p}\n"
                        f"First: ch={params0.nchannels}, sw={params0.sampwidth}, sr={params0.framerate}\n"
                        f"This : ch={params.nchannels}, sw={params.sampwidth}, sr={params.framerate}\n"
                        f"Hint: re-run with --reencode to normalize via ffmpeg."
                    )

            frames_out.append(w.readframes(w.getnframes()))

            # pad after each part except last
            if idx != len(in_paths) - 1:
                pad_ms = pads_ms[idx] if pads_ms else 0
                if pad_ms > 0:
                    pad_frames = int(params.framerate * (pad_ms / 1000.0))
                    frames_out.append(_silence_frames(pad_frames, params.sampwidth, params.nchannels))

    assert params0 is not None
    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(params0.nchannels)
        out.setsampwidth(params0.sampwidth)
        out.setframerate(params0.framerate)
        for fr in frames_out:
            out.writeframes(fr)

def concat_wavs_fixed_pad(in_paths: List[Path], out_path: Path, pad_ms: int = 60) -> None:
    pads = [pad_ms] * max(0, len(in_paths) - 1)
    concat_wavs_with_pads(in_paths, out_path, pads)

# -----------------------------
# Optional: normalize wavs via ffmpeg
# -----------------------------
def ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None

def reencode_wav(in_path: Path, out_path: Path, sr: int = 24000, ch: int = 1) -> None:
    if not ffmpeg_exists():
        raise RuntimeError("ffmpeg not found. Install it or disable --reencode.")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-ac", str(ch),
        "-ar", str(sr),
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# -----------------------------
# Seg config loader
# -----------------------------
def load_seg_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Segmentation config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

# -----------------------------
# Smart segment normalize (supports SegResult or list)
# -----------------------------
def run_smart_segment(text: str, seg_cfg: Union[Path, str, Dict[str, Any]]) -> Tuple[List[str], List[int]]:
    if SEGMENT_TEXT is None:
        raise SystemExit("❌ --smart مطلوب لكن ما قدرنا نحمّل segment_text. تأكد من وجود src/speaker/smart_seg/engine.py وأن src موجود في sys.path.")

    res = SEGMENT_TEXT(text, seg_cfg)  # type: ignore

    # إذا رجّع SegResult
    if hasattr(res, "chunks"):
        chunks = list(getattr(res, "chunks"))
        pads = list(getattr(res, "pad_ms", []))
        return chunks, pads

    # إذا رجّع dict (debug export)
    if isinstance(res, dict) and "chunks" in res:
        chunks = list(res.get("chunks", []))
        pads = list(res.get("pad_ms", []))
        return chunks, pads

    # إذا رجّع list
    if isinstance(res, list):
        return res, []

    # غير معروف
    raise SystemExit("❌ segment_text رجّع نوع غير متوقع. لازم SegResult أو dict فيه chunks أو list[str].")

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="XTTS Arabic TTS CLI (Agent-ready + Smart Segmentation)")

    ap.add_argument("--text", help="Text to synthesize")
    ap.add_argument("--text-file", help="Path to UTF-8 .txt file (relative to project root allowed)")
    ap.add_argument("--out", default="salem_test.wav", help="Output wav name (in outputs/raw)")
    ap.add_argument("--ref", default=str(DEFAULT_REF), help="Reference wav path")
    ap.add_argument("--lang", default="ar", help="Language code (default: ar)")

    ap.add_argument("--clean", dest="clean", action="store_true", help="Clean text for Arabic TTS")
    ap.add_argument("--no-clean", dest="clean", action="store_false", help="Disable cleaning")
    ap.set_defaults(clean=True)

    ap.add_argument("--chatty", action="store_true", help="Apply chatty Saudi-light filter")
    ap.add_argument("--no-starters", action="store_true", help="Disable chatty starters (شوف/طيب...)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for stable starters")

    # CAMeL selective diacritization
    ap.add_argument("--diacritize", action="store_true", help="Selective diacritization via CAMeL Tools (recommended with lexicon)")
    ap.add_argument("--diacritize-words", default="الحجري,الياسي,الكندي,المعني,الشحي,الدرعي,الحبسي",
                    help="Comma-separated trigger words for selective diacritization")

    # Lexicon
    ap.add_argument("--lexicon", action="store_true", help="Apply lexicon corrections (recommended)")
    ap.add_argument("--lexicon-file", default=str(DEFAULT_LEXICON_FILE), help="Path to lexicon JSON file")

    # Smart Segmentation
    ap.add_argument("--smart", action="store_true", help="Use Smart Segmentation Engine (dua/podcast aware)")
    ap.add_argument("--seg-config", default=str(DEFAULT_SEG_CONFIG), help="Path to segmentation JSON config")
    ap.add_argument("--print-segments", action="store_true", help="Print segments after finalize (debug)")

    # Chunking/merge
    ap.add_argument("--split", action="store_true", help="Split into chunks (smart or fallback)")
    ap.add_argument("--max-len", type=int, default=140, help="Max chars per chunk (fallback only)")
    ap.add_argument("--concat", action="store_true", help="After split: merge parts into *_full.wav")
    ap.add_argument("--pad-ms", type=int, default=1000, help="Silence padding between parts when merging (fallback/fixed)")
    ap.add_argument("--keep-parts", action="store_true", help="Keep individual chunk wav files after concatenation (default: delete)")

    ap.add_argument("--reencode", action="store_true", help="Normalize each part via ffmpeg before merging")
    ap.add_argument("--sr", type=int, default=24000, help="Sample rate for reencode (default 24000)")
    ap.add_argument("--ch", type=int, default=1, help="Channels for reencode (default 1)")

    # Auto-Learning Flag
    ap.add_argument("--learn", nargs=2, metavar=("ENGLISH_WORD", "ARABIC_PRONUNCIATION"),
                    help="تضيف كلمة نطق جديدة للقاموس مباشرة (مثال: --learn 'AI' 'إِي آي')")

    args = ap.parse_args()

    # --learn execution block
    if args.learn:
        eng_word, ar_pron = args.learn
        eng_word = eng_word.strip()
        ar_pron = ar_pron.strip()

        lex_file = resolve_path(args.lexicon_file)
        if not lex_file.exists():
            raise FileNotFoundError(f"Lexicon file not found: {lex_file}")

        try:
            data = json.loads(lex_file.read_text(encoding="utf-8"))
            if "names_pronunciation" not in data:
                data["names_pronunciation"] = []

            # Check if exists
            for item in data["names_pronunciation"]:
                if isinstance(item, dict) and item.get("original") == eng_word:
                    print(f"⚠️ الكلمة '{eng_word}' موجودة مسبقاً بنطق '{item.get('formatted')}'.")
                    sys.exit(0)

            # Add and save
            data["names_pronunciation"].append({
                "original": eng_word,
                "formatted": ar_pron
            })
            lex_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"✅ تم بنجاح إضافة الكلمة:")
            print(f"   الكلمة: {eng_word}\n   النطق : {ar_pron}")
            sys.exit(0)
        except Exception as e:
            print(f"❌ حدث خطأ أثناء إضافة الكلمة: {e}")
            sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)

    ref = resolve_path(args.ref)
    if not ref.exists():
        raise FileNotFoundError(f"Reference file not found: {ref}")

    text = load_text(args.text, args.text_file)

    # 1) Lexicon (do this first so English names are converted to Arabic before clean drops them)
    if args.lexicon:
        lex_file = resolve_path(args.lexicon_file)
        lex_map = load_lexicon_map(lex_file)
        text = apply_lexicon(text, lex_map)

    # 1.5) Unknown word check (Runs after known ones are replaced to arabic)
    pending_file = BASE_DIR / "configs" / "pending_words.json"
    check_for_unknown_words(text, pending_file)

    # 2) Clean
    if args.clean:
        text = clean_for_ar_tts(text)

    # 3) Chatty
    if args.chatty:
        text = smart_filter_chatty(text, starters=(not args.no_starters))

    # 4) Selective diacritization (after lexicon/chatty)
    if args.diacritize:
        if _DISAMBIG is None:
            print("⚠️ CAMeL Tools model not ready, skipping --diacritize")
        else:
            triggers = [w.strip() for w in args.diacritize_words.split(",") if w.strip()]
            text = diacritize_selective(text, triggers)

    # Init model once
    if not text.strip():
        print("⚠️ تنبيه: النص المتبقي بعد التنظيف فارغ تماماً. لا يوجد شيء لتحويله إلى صوت.")
        sys.exit(0)

    tts = TTS(MODEL_NAME)
    out_path = OUT_DIR / args.out

    # Split mode => multiple wavs
    if args.split:
        base = out_path.stem

        pads_ms: List[int] = []
        if args.smart:
            seg_cfg = load_seg_config(resolve_path(args.seg_config))
            segments, pads_ms = run_smart_segment(text, seg_cfg)
        else:
            segments = split_to_chunks(text, max_chars=args.max_len)

        segments = finalize_segments(segments)

        # إذا smart رجّع pads، لازم نطابقه مع segments بعد finalize
        # (لو finalize شال مقطع، نخلي fallback pad ثابت)
        if pads_ms and len(pads_ms) != max(0, len(segments) - 1):
            pads_ms = []

        if args.print_segments:
            print("\n--- SEGMENTS (after finalize) ---")
            for i, s in enumerate(segments):
                print(f"[{i:03d}] {s}")
            print("--- END SEGMENTS ---\n")

        if not segments:
            raise ValueError("No usable segments after processing.")

        part_paths: List[Path] = []
        for i, s in enumerate(segments):
            p = OUT_DIR / f"{base}_{i:03d}.wav"
            tts.tts_to_file(
                text=s,
                file_path=str(p),
                speaker_wav=str(ref),
                language=args.lang,
            )
            part_paths.append(p)
            print("Saved:", p)

        if args.concat:
            full = OUT_DIR / f"{base}_full.wav"

            # normalize (optional)
            use_paths = part_paths
            if args.reencode:
                norm_dir = OUT_DIR / f"{base}_norm"
                norm_dir.mkdir(parents=True, exist_ok=True)
                norm_paths: List[Path] = []
                for p in part_paths:
                    np = norm_dir / p.name
                    reencode_wav(p, np, sr=args.sr, ch=args.ch)
                    norm_paths.append(np)
                use_paths = norm_paths

            # merge with smart pads if available else fixed pad
            if pads_ms:
                concat_wavs_with_pads(use_paths, full, pads_ms)
            else:
                concat_wavs_fixed_pad(use_paths, full, pad_ms=args.pad_ms)

            print("Merged:", full)

            # Cleanup parts
            if not args.keep_parts:
                for p in part_paths:
                    if p.exists():
                        p.unlink()
                if args.reencode:
                    if norm_dir.exists():
                        shutil.rmtree(norm_dir)
                print("Cleaned up individual part files.")

        return

    # Single file mode
    tts.tts_to_file(
        text=text,
        file_path=str(out_path),
        speaker_wav=str(ref),
        language=args.lang,
    )
    print("Saved:", out_path)

if __name__ == "__main__":
    main()