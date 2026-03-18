#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json

# ✅ analyzer
from src.brain.project_analyzer import (
    collect_python_files,
    analyze_python_file,
    build_graph,
    detect_issues,
    build_cleanup_candidates,
)

# ✅ modules الأخرى
from src.brain.dependency_mapper import build_dependency_map
from src.brain.architecture_guard import main as architecture_main
from src.brain.restructure_planner import main as restructure_main
from src.brain.auto_cleanup import main as cleanup_exec


OUTPUT_DIR = Path("data/brain")


def save_json(name: str, data):
    path = OUTPUT_DIR / name
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Saved: {path}")


def run_core_analysis():
    print("🧠 STEP 1: Core Analysis\n")

    files = collect_python_files()
    entries = [analyze_python_file(p) for p in files]

    graph = build_graph(entries)
    issues = detect_issues(graph)
    cleanup = build_cleanup_candidates(graph)

    save_json("project_graph.json", graph)
    save_json("issues_detected.json", issues)
    save_json("cleanup_candidates.json", cleanup)

    return graph


def run_dependency(graph):
    print("\n🔗 STEP 2: Dependency Mapping\n")

    dep_map = build_dependency_map(graph)
    save_json("dependency_map.json", dep_map)


def run_architecture():
    print("\n🏗 STEP 3: Architecture Check\n")
    architecture_main()


def run_restructure():
    print("\n📐 STEP 4: Restructure Plan\n")
    restructure_main()


def run_cleanup():
    print("\n🧹 STEP 5: Cleanup Execution\n")
    cleanup_exec()


def main():
    print("🚀 FULL PROJECT BRAIN STARTED\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    graph = run_core_analysis()
    run_dependency(graph)
    run_architecture()
    run_restructure()

    print("\n🎉 Brain analysis completed!")

    print("\n💡 إذا تريد تنفيذ التنظيف:")
    print("👉 شغل: python brain.py --fix")


if __name__ == "__main__":
    import sys

    if "--fix" in sys.argv:
        run_cleanup()
    else:
        main()