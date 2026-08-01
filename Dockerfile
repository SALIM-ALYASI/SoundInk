# SoundInk (APVA Next) - Production image
FROM python:3.11-slim

# تبعيات نظام مطلوبة لمعالجة الصوت (ffmpeg لـpydub, libsndfile لـsoundfile,
# espeak-ng لبعض مكتبات تحليل النطق اللي يستخدمها Coqui TTS).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# نسخة CPU فقط من PyTorch (سيرفر بدون GPU) لتفادي تحميل مكتبات CUDA الضخمة غير المستخدمة
RUN pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# تفعيل الموافقة على ترخيص نموذج XTTS تلقائيًا (يتطلبه Coqui TTS عند أول تحميل).
ENV COQUI_TOS_AGREED=1
ENV PYTHONUNBUFFERED=1

EXPOSE 5050

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5050"]
