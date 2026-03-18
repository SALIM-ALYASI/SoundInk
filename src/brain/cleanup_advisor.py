from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_FILE = PROJECT_ROOT / "data" / "brain" / "project_graph.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "brain" / "cleanup_advisor_report.json"


CORE_KEEP = (
    "app.py",
    "api/",
    "core/",
    "services/",
)


def load_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def classify_module(mod: dict) -> dict:
    path = mod["path"]

    if path == "app.py" or path.startswith(CORE_KEEP):
        return {"path": path, "decision": "keep", "reason": "core_runtime"}

    if not mod.get("functions") and not mod.get("classes") and mod.get("lines", 0) <= 5:
        return {"path": path, "decision": "review", "reason": "empty_placeholder"}

    if "test" in path.lower() or "legacy" in path.lower() or "old" in path.lower():
        return {"path": path, "decision": "review", "reason": "suspicious_name"}

    return {"path": path, "decision": "keep", "reason": "active_or_in_progress"}


def main() -> None:
    graph = load_json(GRAPH_FILE)
    modules = graph.get("modules", [])

    decisions = [classify_module(mod) for mod in modules]

    report = {
        "summary": {
            "total": len(decisions),
            "keep": sum(1 for d in decisions if d["decision"] == "keep"),
            "review": sum(1 for d in decisions if d["decision"] == "review"),
        },
        "decisions": decisions,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Cleanup advisor report saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()