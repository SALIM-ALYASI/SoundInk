from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_FILE = PROJECT_ROOT / "data" / "brain" / "project_graph.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "brain" / "dependency_map.json"


def load_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_dependency_map(graph: dict) -> dict:
    modules = graph.get("modules", [])
    result = {
        "nodes": [],
        "edges": [],
    }

    for mod in modules:
        module_name = mod["module"]
        result["nodes"].append({
            "module": module_name,
            "path": mod["path"],
        })

        for imp in mod.get("local_imports", []):
            result["edges"].append({
                "from": module_name,
                "to": imp,
                "type": "local_import",
            })

    return result


def main() -> None:
    graph = load_json(GRAPH_FILE)
    dep_map = build_dependency_map(graph)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(dep_map, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Dependency map saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()