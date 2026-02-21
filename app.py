import os
import subprocess
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

# مسارات المشروع
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "raw"
INPUT_FILE = BASE_DIR / "input.txt"

# Ensure output dir exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/synthesize", methods=["POST"])
def synthesize():
    data = request.json
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "النص المدخل فارغ!"}), 400

    try:
        # 1. اكتب النص في ملف input.txt
        INPUT_FILE.write_text(text, encoding="utf-8")

        # 2. قم بتشغيل سكربت الصوت في الخلفية
        # نستخدم نفس الأوامر من run_audio.sh
        cmd = [
            "python", "apva.py",
            "--text-file", "input.txt",
            "--out", "final_audio.wav",
            "--split", "--concat", "--lexicon"
        ]
        
        # شغل العملية وانتظر النتيجة
        process = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)

        if process.returncode != 0:
            print("TTS Error Log:", process.stderr)
            return jsonify({"error": "حدث خطأ أثناء توليد الصوت. راجع الـ Console."}), 500

        # بناء اسم الملف المخرج (بسبب دمج المقاطع يكون اسمه final_audio_full.wav)
        output_file = OUTPUT_DIR / "final_audio_full.wav"
        
        if not output_file.exists():
            return jsonify({"error": "تم توليد الصوت لكن الملف مفقود!"}), 500

        return jsonify({
            "message": "تم التوليد بنجاح!",
            "audio_url": "/api/audio"
        })

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/audio")
def get_audio():
    audio_path = OUTPUT_DIR / "final_audio_full.wav"
    if not audio_path.exists():
        return "Audio not found", 404
    # no-cache لضمان تحميل النسخة الأحدث دائماً
    response = send_file(audio_path, mimetype="audio/wav")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
