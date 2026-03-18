from __future__ import annotations

import ast
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCAN_DIRS = [
    "api",
    "core",
    "services",
    "src",
]

SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "_archive",
}

OUTPUT_GRAPH = PROJECT_ROOT / "data/brain/project_graph.json"
OUTPUT_ISSUES = PROJECT_ROOT / "data/brain/issues_detected.json"
OUTPUT_CANDIDATES = PROJECT_ROOT / "data/brain/cleanup_candidates.json"


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def module_name_from_path(path: Path) -> str:
    rel = path.relative_to(PROJECT_ROOT)
    no_suffix = rel.with_suffix("")
    return ".".join(no_suffix.parts)


def analyze_python_file(path: Path) -> dict:
    result = {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "module": module_name_from_path(path),
        "imports": [],
        "functions": [],
        "classes": [],
        "lines": 0,
        "errors": [],
    }

    try:
        content = path.read_text(encoding="utf-8")
        result["lines"] = len(content.splitlines())
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result["imports"].append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                result["imports"].append(mod)

            elif isinstance(node, ast.FunctionDef):
                result["functions"].append(node.name)

            elif isinstance(node, ast.AsyncFunctionDef):
                result["functions"].append(node.name)

            elif isinstance(node, ast.ClassDef):
                result["classes"].append(node.name)

    except Exception as exc:
        result["errors"].append(str(exc))

    result["imports"] = sorted(set(x for x in result["imports"] if x))
    result["functions"] = sorted(set(result["functions"]))
    result["classes"] = sorted(set(result["classes"]))

    return result


def collect_python_files() -> list[Path]:
    files: list[Path] = []

    for folder in SCAN_DIRS:
        base = PROJECT_ROOT / folder
        if not base.exists():
            continue

        for path in base.rglob("*.py"):
            if should_skip(path):
                continue
            files.append(path)

    app_file = PROJECT_ROOT / "app.py"
    if app_file.exists():
        files.append(app_file)

    return sorted(set(files))


def build_graph(entries: list[dict]) -> dict:
    modules = {entry["module"] for entry in entries}

    graph = {
        "project_root": str(PROJECT_ROOT),
        "modules": [],
    }

    for entry in entries:
        local_imports = []
        external_imports = []

        for imp in entry["imports"]:
            if imp in modules or any(m.startswith(f"{imp}.") for m in modules):
                local_imports.append(imp)
            else:
                external_imports.append(imp)

        graph["modules"].append({
            "path": entry["path"],
            "module": entry["module"],
            "functions": entry["functions"],
            "classes": entry["classes"],
            "lines": entry["lines"],
            "local_imports": sorted(set(local_imports)),
            "external_imports": sorted(set(external_imports)),
            "errors": entry["errors"],
        })

    return graph


def detect_issues(graph: dict) -> dict:
    issues = {
        "parse_errors": [],
        "empty_modules": [],
        "suspicious_modules": [],
    }

    for mod in graph["modules"]:
        if mod["errors"]:
            issues["parse_errors"].append({
                "path": mod["path"],
                "errors": mod["errors"],
            })

        if not mod["functions"] and not mod["classes"] and mod["lines"] <= 5:
            issues["empty_modules"].append(mod["path"])

        suspicious_names = (
            "test",
            "old",
            "legacy",
            "backup",
            "tmp",
            "experimental",
        )
        if any(word in mod["path"].lower() for word in suspicious_names):
            issues["suspicious_modules"].append(mod["path"])

    return issues


def build_cleanup_candidates(graph: dict) -> dict:
    candidates = {
        "review": [],
        "keep": [],
    }

    for mod in graph["modules"]:
        path = mod["path"]

        if path == "app.py":
            candidates["keep"].append(path)
            continue

        if path.startswith(("api/", "core/", "services/")):
            candidates["keep"].append(path)
            continue

        if path.endswith(("routes.py", "schemas.py", "processor.py", "inference_manager.py")):
            candidates["keep"].append(path)
            continue

        if not mod["functions"] and not mod["classes"] and mod["lines"] <= 5:
            candidates["review"].append({
                "path": path,
                "reason": "empty_or_placeholder",
            })
        else:
            candidates["keep"].append(path)

    return candidates


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    files = collect_python_files()
    entries = [analyze_python_file(path) for path in files]

    graph = build_graph(entries)
    issues = detect_issues(graph)
    cleanup = build_cleanup_candidates(graph)

    write_json(OUTPUT_GRAPH, graph)
    write_json(OUTPUT_ISSUES, issues)
    write_json(OUTPUT_CANDIDATES, cleanup)

    print("Done.")
    print(f"Graph saved to: {OUTPUT_GRAPH}")
    print(f"Issues saved to: {OUTPUT_ISSUES}")
    print(f"Cleanup saved to: {OUTPUT_CANDIDATES}")


if __name__ == "__main__":
    main()