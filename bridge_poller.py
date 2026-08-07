#!/usr/bin/env python3
"""
جسر بين السيرفر (alyasi.dev VPS) وSoundInk المحلي.

السيرفر يحط طلب توليد صوت بمجلد narration_queue/pending عبر newsVideo.js
(بدل ما يشغّل SoundInk بنفسه ويسبب OOM). هذا السكربت يراقب ذاك المجلد عبر
SSH كل POLL_INTERVAL ثانية، ولو لقى طلب جديد:
  1. يجيب نص الطلب.
  2. يستدعي SoundInk المحلي (لازم يكون شغّال على localhost:5050).
  3. يرفع ملف الصوت الناتج لمجلد done بالسيرفر.
  4. يمسح ملف الطلب المعلّق.

شغّله وخلّه شغّال بالخلفية (أو أضفه كـLaunchAgent يشتغل مع بداية الجهاز).
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

SERVER = "root@167.233.163.230"
REMOTE_BASE = "/srv/apps/newsbot/data/narration_queue"
LOCAL_SOUNDINK_URL = "http://localhost:5050/api/v1/speak"
POLL_INTERVAL_SEC = 20
SSH_TIMEOUT_SEC = 20


def run(cmd: list[str], timeout: int = SSH_TIMEOUT_SEC) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def list_pending() -> list[str]:
    res = run(["ssh", SERVER, f"ls {REMOTE_BASE}/pending/ 2>/dev/null"])
    if res.returncode != 0:
        return []
    return [line.strip() for line in res.stdout.splitlines() if line.strip().endswith(".json")]


def fetch_job(filename: str) -> dict | None:
    res = run(["ssh", SERVER, f"cat {REMOTE_BASE}/pending/{filename}"])
    if res.returncode != 0 or not res.stdout.strip():
        return None
    try:
        return json.loads(res.stdout)
    except json.JSONDecodeError:
        return None


def synthesize(text: str, voice_id: str, out_path: Path) -> None:
    resp = requests.post(
        LOCAL_SOUNDINK_URL,
        json={"text": text, "voice_id": voice_id, "preview": True},
        timeout=600,
    )
    resp.raise_for_status()
    out_path.write_bytes(resp.content)


def upload_result(job_id: str, local_audio_path: Path) -> None:
    # نرفع لاسم مؤقت أولاً، وبعدين نعيد تسميته بأمر منفصل عبر ssh - عشان
    # السيرفر ما يشوف الملف النهائي إلا وهو مكتمل بالكامل. لو نرفع على
    # الاسم النهائي مباشرة، ممكن newsVideo.js يلقطه بمنتصف النقل (fs.existsSync
    # ترجع true بينما scp لسا يكتب) ويقرأ ملف صوت تالف.
    final_path = f"{REMOTE_BASE}/done/{job_id}.wav"
    tmp_path = f"{final_path}.tmp"
    res = run(["scp", str(local_audio_path), f"{SERVER}:{tmp_path}"], timeout=120)
    if res.returncode != 0:
        raise RuntimeError(f"scp فشل: {res.stderr.strip()}")
    res = run(["ssh", SERVER, f"mv {tmp_path} {final_path}"], timeout=SSH_TIMEOUT_SEC)
    if res.returncode != 0:
        raise RuntimeError(f"إعادة التسمية بالسيرفر فشلت: {res.stderr.strip()}")


def report_failure(job_id: str, error: str) -> None:
    payload = json.dumps({"error": error}, ensure_ascii=False)
    subprocess.run(
        ["ssh", SERVER, f"cat > {REMOTE_BASE}/failed/{job_id}.json"],
        input=payload,
        text=True,
        timeout=SSH_TIMEOUT_SEC,
    )


def remove_pending(filename: str) -> None:
    run(["ssh", SERVER, f"rm -f {REMOTE_BASE}/pending/{filename}"])


def process_job(filename: str) -> None:
    job_id = filename.rsplit(".", 1)[0]
    job = fetch_job(filename)
    if job is None:
        print(f"[{job_id}] تعذّر قراءة الطلب، تجاهله.")
        remove_pending(filename)
        return

    text = job.get("text", "")
    voice_id = job.get("voice_id", "salem_base")
    print(f"[{job_id}] توليد صوت ({len(text)} حرف)...")

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / f"{job_id}.wav"
        try:
            synthesize(text, voice_id, out_path)
            upload_result(job_id, out_path)
            remove_pending(filename)
            print(f"[{job_id}] تم ورُفع بنجاح.")
        except Exception as exc:  # noqa: BLE001
            print(f"[{job_id}] فشل: {exc}")
            try:
                report_failure(job_id, str(exc))
            except Exception:  # noqa: BLE001
                pass
            remove_pending(filename)


def main() -> None:
    print("🌉 بدأ جسر SoundInk المحلي - يراقب طلبات السيرفر كل", POLL_INTERVAL_SEC, "ثانية")
    # مهلة بداية بسيطة - عند إقلاع الجهاز، SoundInk المحلي (uvicorn) قد يكون
    # لسا يبدأ، فما نبدأ نعالج طلبات لين يجهز.
    time.sleep(15)
    while True:
        try:
            for filename in list_pending():
                process_job(filename)
        except Exception as exc:  # noqa: BLE001
            print("⚠️ خطأ بدورة المراقبة:", exc, file=sys.stderr)
        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
