from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LEXICON_DIR = PROJECT_ROOT / "data" / "lexicon"
LEXICON_FILE = LEXICON_DIR / "pronunciation_lexicon.json"
SUGGESTIONS_FILE = LEXICON_DIR / "lexicon_suggestions.json"
PREVIEWS_DIR = LEXICON_DIR / "previews"


DEFAULT_LEXICON = {
    "misc_pronunciation": [],
    "names_pronunciation": [],
    "tribes_pronunciation": [],
}

VALID_STATUSES = {"pending", "approved", "rejected", "needs_edit"}
DEFAULT_CATEGORY = "misc_pronunciation"

ARABIC_DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")
MULTI_SPACE_RE = re.compile(r"\s+")
UNKNOWN_WORD_SUFFIX_RE = re.compile(r"\s*:\s*Unknown word\.?\s*$", re.IGNORECASE)

MIN_SUGGESTION_SIMILARITY = 0.35
AUTO_REJECT_EMPTY_RECOGNIZED = False
MAX_SUGGESTION_WORDS = 3


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: str | None) -> str:
    return (value or "").strip()


def remove_diacritics(text: str | None) -> str:
    return ARABIC_DIACRITICS_RE.sub("", text or "")


def clean_text_field(text: str | None) -> str:
    text = str(text or "").strip()
    text = UNKNOWN_WORD_SUFFIX_RE.sub("", text)

    # إذا كانت هناك رسالة أو جزء زائد بعد :
    if ":" in text:
        text = text.split(":", 1)[0]

    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def clean_suggestion_text(text: str | None) -> str:
    return clean_text_field(text)


def normalize_compare_text(text: str | None) -> str:
    text = clean_suggestion_text(text)
    text = remove_diacritics(text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def normalize_category(category: str | None) -> str:
    category = clean_text(category)
    return category or DEFAULT_CATEGORY


def word_count(text: str | None) -> int:
    cleaned = clean_suggestion_text(text)
    if not cleaned:
        return 0
    return len(cleaned.split())


def calc_text_similarity(a: str | None, b: str | None) -> float:
    aa = normalize_compare_text(a)
    bb = normalize_compare_text(b)

    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0

    return round(SequenceMatcher(None, aa, bb).ratio(), 4)


def is_low_quality_suggestion(
    original: str,
    suggested: str,
    recognized: str,
    similarity: float,
) -> bool:
    original_n = normalize_compare_text(original)
    suggested_n = normalize_compare_text(suggested)
    recognized_n = normalize_compare_text(recognized)

    if not original_n or not suggested_n:
        return True

    if len(original_n) < 2 or len(suggested_n) < 2:
        return True

    if word_count(original) > MAX_SUGGESTION_WORDS:
        return True

    if word_count(suggested) > MAX_SUGGESTION_WORDS:
        return True

    if similarity < MIN_SUGGESTION_SIMILARITY:
        return True

    if AUTO_REJECT_EMPTY_RECOGNIZED and not recognized_n:
        return True

    return False


def ensure_storage() -> None:
    LEXICON_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    if not LEXICON_FILE.exists():
        LEXICON_FILE.write_text(
            json.dumps(DEFAULT_LEXICON, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if not SUGGESTIONS_FILE.exists():
        SUGGESTIONS_FILE.write_text("[]", encoding="utf-8")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_lexicon() -> dict[str, list[dict[str, Any]]]:
    ensure_storage()

    data = load_json(LEXICON_FILE, DEFAULT_LEXICON)
    if not isinstance(data, dict):
        data = {}

    normalized: dict[str, list[dict[str, Any]]] = {}

    for key, default_value in DEFAULT_LEXICON.items():
        value = data.get(key, default_value)
        normalized[key] = value if isinstance(value, list) else []

    for key, value in data.items():
        if key not in normalized and isinstance(value, list):
            normalized[key] = value

    return normalized


def save_lexicon(data: dict[str, Any]) -> None:
    ensure_storage()
    save_json(LEXICON_FILE, data)


def load_suggestions() -> list[dict[str, Any]]:
    ensure_storage()

    data = load_json(SUGGESTIONS_FILE, [])
    if not isinstance(data, list):
        return []

    clean_items: list[dict[str, Any]] = []
    changed = False

    for item in data:
        if not isinstance(item, dict):
            changed = True
            continue

        cleaned = dict(item)
        cleaned["original"] = clean_suggestion_text(cleaned.get("original"))
        cleaned["suggested"] = clean_suggestion_text(cleaned.get("suggested"))
        cleaned["recognized"] = clean_suggestion_text(cleaned.get("recognized"))
        cleaned["category"] = normalize_category(cleaned.get("category"))
        cleaned["count"] = int(cleaned.get("count", 1) or 1)

        status = clean_text(cleaned.get("status"))
        if status not in VALID_STATUSES:
            cleaned["status"] = "pending"

        if not cleaned["original"] or not cleaned["suggested"]:
            changed = True
            continue

        if cleaned != item:
            changed = True

        clean_items.append(cleaned)

    if changed:
        save_suggestions(clean_items)

    return clean_items


def save_suggestions(data: list[dict[str, Any]]) -> None:
    ensure_storage()

    clean_items: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        cleaned = dict(item)
        cleaned["original"] = clean_suggestion_text(cleaned.get("original"))
        cleaned["suggested"] = clean_suggestion_text(cleaned.get("suggested"))
        cleaned["recognized"] = clean_suggestion_text(cleaned.get("recognized"))
        cleaned["category"] = normalize_category(cleaned.get("category"))
        cleaned["count"] = int(cleaned.get("count", 1) or 1)

        status = clean_text(cleaned.get("status"))
        cleaned["status"] = status if status in VALID_STATUSES else "pending"

        if not cleaned["original"] or not cleaned["suggested"]:
            continue

        clean_items.append(cleaned)

    save_json(SUGGESTIONS_FILE, clean_items)


def get_suggestion(suggestion_id: str) -> dict[str, Any] | None:
    suggestion_id = clean_text(suggestion_id)
    if not suggestion_id:
        return None

    for item in load_suggestions():
        if item.get("id") == suggestion_id:
            return item

    return None


def find_duplicate_suggestion(
    original: str,
    suggested: str,
    category: str,
) -> dict[str, Any] | None:
    original_n = normalize_compare_text(original)
    suggested_n = normalize_compare_text(suggested)
    category = normalize_category(category)

    for item in load_suggestions():
        item_original_n = normalize_compare_text(item.get("original"))
        item_suggested_n = normalize_compare_text(item.get("suggested"))
        item_category = normalize_category(item.get("category"))

        if (
            item_original_n == original_n
            and item_suggested_n == suggested_n
            and item_category == category
            and item.get("status") in {"pending", "needs_edit", "approved"}
        ):
            return item

    return None


def smart_create_suggestion(
    original: str,
    suggested: str,
    category: str = DEFAULT_CATEGORY,
    recognized: str = "",
    similarity: float = 0.0,
    audio_path: str | None = None,
) -> dict[str, Any]:
    ensure_storage()

    original = clean_suggestion_text(original)
    suggested = clean_suggestion_text(suggested)
    recognized = clean_suggestion_text(recognized)
    category = normalize_category(category)
    audio_path = clean_text(audio_path)

    if not similarity:
        similarity = calc_text_similarity(suggested or original, recognized or original)

    similarity = round(float(similarity), 4)

    if is_low_quality_suggestion(
        original=original,
        suggested=suggested,
        recognized=recognized,
        similarity=similarity,
    ):
        raise ValueError("Suggestion quality is too low")

    duplicate = find_duplicate_suggestion(
        original=original,
        suggested=suggested,
        category=category,
    )

    if duplicate:
        suggestions = load_suggestions()
        for item in suggestions:
            if item.get("id") == duplicate.get("id"):
                item["count"] = int(item.get("count", 1)) + 1
                item["recognized"] = recognized or item.get("recognized", "")
                item["similarity"] = max(
                    round(float(item.get("similarity", 0.0)), 4),
                    similarity,
                )
                if audio_path:
                    item["audio_path"] = audio_path
                item["updated_at"] = utc_now_iso()
                save_suggestions(suggestions)
                return item
        return duplicate

    now = utc_now_iso()

    item = {
        "id": f"sug_{uuid4().hex[:12]}",
        "original": original,
        "suggested": suggested,
        "recognized": recognized,
        "category": category,
        "status": "pending",
        "similarity": similarity,
        "audio_path": audio_path,
        "count": 1,
        "created_at": now,
        "updated_at": now,
    }

    suggestions = load_suggestions()
    suggestions.append(item)
    save_suggestions(suggestions)
    return item


def create_suggestion(
    original: str,
    suggested: str,
    category: str = DEFAULT_CATEGORY,
    recognized: str = "",
    similarity: float = 0.0,
    audio_path: str | None = None,
) -> dict[str, Any]:
    return smart_create_suggestion(
        original=original,
        suggested=suggested,
        category=category,
        recognized=recognized,
        similarity=similarity,
        audio_path=audio_path,
    )


def update_suggestion(
    suggestion_id: str,
    *,
    suggested: str | None = None,
    status: str | None = None,
    audio_path: str | None = None,
) -> dict[str, Any] | None:
    suggestion_id = clean_text(suggestion_id)
    if not suggestion_id:
        return None

    suggestions = load_suggestions()

    for item in suggestions:
        if item.get("id") != suggestion_id:
            continue

        if suggested is not None:
            cleaned_suggested = clean_suggestion_text(suggested)
            if not cleaned_suggested:
                raise ValueError("suggested cannot be empty")
            item["suggested"] = cleaned_suggested

        if status is not None:
            if status not in VALID_STATUSES:
                raise ValueError(f"Invalid status: {status}")
            item["status"] = status

        if audio_path is not None:
            item["audio_path"] = clean_text(audio_path)

        item["updated_at"] = utc_now_iso()
        save_suggestions(suggestions)
        return item

    return None


def add_lexicon_entry(
    category: str,
    original: str,
    formatted: str,
    note: str | None = None,
) -> dict[str, Any]:
    category = normalize_category(category)
    original = clean_suggestion_text(original)
    formatted = clean_suggestion_text(formatted)
    note = clean_text(note)

    if not original:
        raise ValueError("original is required")

    if not formatted:
        raise ValueError("formatted is required")

    data = load_lexicon()

    if category not in data:
        data[category] = []

    for item in data[category]:
        if normalize_compare_text(item.get("original")) == normalize_compare_text(original):
            item["formatted"] = formatted
            if note:
                item["note"] = note
            save_lexicon(data)
            return item

    entry: dict[str, Any] = {
        "original": original,
        "formatted": formatted,
    }

    if note:
        entry["note"] = note

    data[category].append(entry)
    save_lexicon(data)
    return entry


def approve_suggestion(suggestion_id: str) -> dict[str, Any] | None:
    item = get_suggestion(suggestion_id)
    if not item:
        return None

    original = clean_suggestion_text(item.get("original"))
    suggested = clean_suggestion_text(item.get("suggested"))
    category = normalize_category(item.get("category"))

    if not original or not suggested:
        raise ValueError("Suggestion is missing original or suggested value")

    add_lexicon_entry(
        category=category,
        original=original,
        formatted=suggested,
        note="approved from suggestions",
    )

    return update_suggestion(suggestion_id, status="approved")


def reject_suggestion(suggestion_id: str) -> dict[str, Any] | None:
    return update_suggestion(suggestion_id, status="rejected")


def mark_suggestion_needs_edit(
    suggestion_id: str,
    suggested: str,
) -> dict[str, Any] | None:
    return update_suggestion(
        suggestion_id,
        suggested=suggested,
        status="needs_edit",
    )


def grouped_suggestions() -> dict[str, list[dict[str, Any]]]:
    data: dict[str, list[dict[str, Any]]] = {
        "pending": [],
        "approved": [],
        "rejected": [],
        "needs_edit": [],
    }

    for item in load_suggestions():
        status = item.get("status", "pending")
        if status not in data:
            status = "pending"
        data[status].append(item)

    for key in data:
        data[key] = sorted(
            data[key],
            key=lambda x: x.get("updated_at", ""),
            reverse=True,
        )

    return data