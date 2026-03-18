from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEMORY_FILE = PROJECT_ROOT / "data" / "brain" / "project_memory.json"


def load_memory() -> dict:
    if not MEMORY_FILE.exists():
        return {
            "created_at": datetime.utcnow().isoformat(),
            "notes": [],
            "decisions": [],
            "known_issues": [],
        }

    return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))


def save_memory(memory: dict) -> None:
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(
        json.dumps(memory, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def add_note(text: str) -> None:
    memory = load_memory()
    memory["notes"].append({
        "text": text,
        "at": datetime.utcnow().isoformat(),
    })
    save_memory(memory)


def add_decision(title: str, details: str) -> None:
    memory = load_memory()
    memory["decisions"].append({
        "title": title,
        "details": details,
        "at": datetime.utcnow().isoformat(),
    })
    save_memory(memory)


def add_issue(issue: str) -> None:
    memory = load_memory()
    memory["known_issues"].append({
        "issue": issue,
        "at": datetime.utcnow().isoformat(),
    })
    save_memory(memory)