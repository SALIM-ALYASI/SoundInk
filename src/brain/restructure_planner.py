from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_FILE = PROJECT_ROOT / "data" / "brain" / "project_graph.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "brain" / "restructure_plan.json"


def load_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def suggest_target(path: str) -> str | None:
    lower = path.lower()

    if "helper" in lower or "utils" in lower:
        return "src/utils/"

    if "memory" in lower:
        return "src/memory/"

    if "advisor" in lower or "analyzer" in lower or "mapper" in lower:
        return "src/brain/"

    if "transcriber" in lower or "comparator" in lower:
        return "src/asr/"

    if "normalizer" in lower:
        return "src/speaker/"

    return None


def main() -> None:
    graph = load_json(GRAPH_FILE)
    modules = graph.get("modules", [])

    moves = []
    keep = []

    for mod in modules:
        path = mod["path"]

        if path == "app.py":
            keep.append(path)
            continue

        target = suggest_target(path)
        if target:
            moves.append({
                "path": path,
                "suggested_target_dir": target,
                "reason": "structure_alignment",
            })
        else:
            keep.append(path)

    report = {
        "summary": {
            "moves": len(moves),
            "keep": len(keep),
        },
        "moves": moves,
        "keep": keep,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Restructure plan saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()