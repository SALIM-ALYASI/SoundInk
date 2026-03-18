from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_FILE = PROJECT_ROOT / "data" / "brain" / "project_graph.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "brain" / "architecture_report.json"


LAYER_RULES = {
    "app": {"can_import": ["api", "core"]},
    "api": {"can_import": ["core"]},
    "core": {"can_import": []},
    "services": {"can_import": ["core"]},
    "src": {"can_import": ["core"]},
}


def load_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def detect_layer(module_path: str) -> str:
    if module_path == "app.py":
        return "app"
    top = module_path.split("/", 1)[0]
    if top in {"api", "core", "services", "src"}:
        return top
    return "other"


def import_layer(import_name: str) -> str:
    top = import_name.split(".", 1)[0]
    if top in {"api", "core", "services", "src"}:
        return top
    return "external"


def main() -> None:
    graph = load_json(GRAPH_FILE)
    modules = graph.get("modules", [])

    violations = []
    summary = {
        "total_modules": len(modules),
        "violations": 0,
    }

    for mod in modules:
        path = mod["path"]
        layer = detect_layer(path)

        if layer not in LAYER_RULES:
            continue

        allowed = set(LAYER_RULES[layer]["can_import"])

        for imp in mod.get("local_imports", []):
            imp_layer = import_layer(imp)

            if imp_layer == "external":
                continue

            if imp_layer not in allowed:
                violations.append({
                    "path": path,
                    "module_layer": layer,
                    "import": imp,
                    "import_layer": imp_layer,
                    "reason": "layer_violation",
                })

    summary["violations"] = len(violations)

    report = {
        "summary": summary,
        "rules": LAYER_RULES,
        "violations": violations,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Architecture report saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()