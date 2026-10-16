#!/usr/bin/env python3
"""
مزامنة قاموس النطق: قاعدة بيانات الناشر بالسيرفر -> SoundInk المحلي.

نفس فلسفة bridge_poller.py (SSH بدل اتصال Postgres مباشر -- القاعدة غير
مكشوفة للإنترنت أصلاً، فقط بين حاويات Docker بالسيرفر). يفحص كل
POLL_INTERVAL_SEC جدول publisher.voice_dictionary_entries عبر SSH+psql
بحثاً عن أي إدخال أحدث من آخر id عولج محلياً، ويضيفه لقاموس SoundInk
المحلي عبر /api/v1/lexicon/add. حالة "آخر id" تُحفظ بملف محلي بسيط
(sync_state.json) -- مو بقاعدة السيرفر، عشان ما نحتاج نضيف عمود جديد
هناك لمجرد تتبع حالة العميل المحلي.
"""

import os
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

SERVER = "home"
POSTGRES_CONTAINER = "alyasi-n8n-postgres-1"
DB_USER = "alyasi_publisher"
DB_NAME = "alyasi_publisher"

LOCAL_SOUNDINK_URL = "http://localhost:5050/api/v1/lexicon/add"
# المفتاح من البيئة (.env) -- ما ينكتب بالكود لأن المستودع عام.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

LOCAL_API_KEY = os.getenv("SOUNDINK_API_KEY") or os.getenv("API_KEY", "")

STATE_FILE = Path(__file__).parent / "data" / "dictionary_sync_state.json"
POLL_INTERVAL_SEC = 120
SSH_TIMEOUT_SEC = 30


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"last_entry_id": 0}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state))


def fetch_new_entries(after_id: int) -> list[dict]:
    # الاستعلام يُمرَّر عبر stdin لا كوسيط -c -- يتفادى مشاكل تهريب الأقواس
    # والمسافات عبر طبقتين من الصدفة (المحلية ثم صدفة السيرفر البعيد عبر ssh).
    query = (
        f"SELECT json_agg(row_to_json(e)) FROM ("
        f"SELECT id, entry_text, normalized_text FROM publisher.voice_dictionary_entries "
        f"WHERE id > {after_id} ORDER BY id"
        f") e;"
    )
    res = subprocess.run(
        ["ssh", SERVER, "docker", "exec", "-i", POSTGRES_CONTAINER,
         "psql", "-U", DB_USER, "-d", DB_NAME, "-t", "-A"],
        input=query, capture_output=True, text=True, timeout=SSH_TIMEOUT_SEC,
    )
    if res.returncode != 0:
        print(f"[sync] SSH/psql failed: {res.stderr.strip()}", file=sys.stderr)
        return []
    output = res.stdout.strip()
    if not output or output == "":
        return []
    try:
        return json.loads(output) or []
    except json.JSONDecodeError:
        print(f"[sync] could not parse psql output: {output[:200]!r}", file=sys.stderr)
        return []


def push_to_local_lexicon(entry_text: str) -> bool:
    try:
        resp = requests.post(
            LOCAL_SOUNDINK_URL,
            headers={"X-API-Key": LOCAL_API_KEY},
            params={"original": entry_text, "formatted": entry_text, "category": "misc_pronunciation"},
            timeout=15,
        )
    except requests.RequestException as exc:
        print(f"[sync] local SoundInk unreachable: {exc}", file=sys.stderr)
        return False

    if resp.status_code == 200:
        return True
    if resp.status_code == 400 and "already exists" in resp.text:
        return True  # مزامن مسبقاً، مو خطأ
    print(f"[sync] local add failed ({resp.status_code}): {resp.text[:200]}", file=sys.stderr)
    return False


def run_once() -> None:
    state = load_state()
    entries = fetch_new_entries(state["last_entry_id"])
    if not entries:
        return

    max_id = state["last_entry_id"]
    for entry in entries:
        ok = push_to_local_lexicon(entry["entry_text"])
        if ok:
            print(f"[sync] synced entry #{entry['id']}: {entry['entry_text']}")
            max_id = max(max_id, entry["id"])
        else:
            # نتوقف عند أول فشل ونعيد المحاولة الدورة الجاية -- بدون
            # تخطي عناصر لو SoundInk المحلي متعطل مؤقتاً (جهاز نايم، إلخ).
            break

    if max_id != state["last_entry_id"]:
        state["last_entry_id"] = max_id
        save_state(state)


def main() -> None:
    print(f"[sync] dictionary_sync started, polling every {POLL_INTERVAL_SEC}s")
    while True:
        try:
            run_once()
        except Exception as exc:
            print(f"[sync] cycle error: {exc}", file=sys.stderr)
        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
