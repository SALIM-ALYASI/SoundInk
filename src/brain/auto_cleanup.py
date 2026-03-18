from pathlib import Path
import json
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRAIN_DIR = PROJECT_ROOT / "data" / "brain"
ARCHIVE_DIR = PROJECT_ROOT / "_archive"


def load_cleanup():
    path = BRAIN_DIR / "cleanup_advisor_report.json"
    if not path.exists():
        return None

    return json.loads(path.read_text())


def archive_file(path):
    src = PROJECT_ROOT / path

    if not src.exists():
        return

    dst = ARCHIVE_DIR / path
    dst.parent.mkdir(parents=True, exist_ok=True)

    shutil.move(str(src), str(dst))
    print("Archived:", path)


def main():

    report = load_cleanup()

    if not report:
        print("No cleanup report found.")
        return

    ARCHIVE_DIR.mkdir(exist_ok=True)

    for item in report["decisions"]:

        if item["decision"] != "review":
            continue

        archive_file(item["path"])

    print("Cleanup completed.")


if __name__ == "__main__":
    main()