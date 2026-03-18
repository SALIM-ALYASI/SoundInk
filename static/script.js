document.addEventListener("DOMContentLoaded", () => {
    const MAX_TEXT_LENGTH = 20000;
    const PREVIEW_TEXT_LIMIT = 220;

    const generateBtn = document.getElementById("generate-btn");
    const textInput = document.getElementById("text-input");
    const voiceSelect = document.getElementById("voice-select");
    const btnText = document.querySelector(".btn-text");
    const previewBtn = document.getElementById("preview-btn");
    const previewPlayer = document.getElementById("preview-player");
    const charCount = document.getElementById("char-count");
    const resultSection = document.getElementById("result-section");
    const normalPlayer = document.getElementById("normal-player");
    const errorMsg = document.getElementById("error-message");
    const progressPanel = document.getElementById("progress-panel");
    const progressFill = document.getElementById("progress-fill");
    const progressStatus = document.getElementById("progress-status");
    const progressTimer = document.getElementById("progress-timer");
    const refreshVoicesBtn = document.getElementById("refresh-voices-btn");
    const voiceStatus = document.getElementById("voice-status");
    const apiStatus = document.getElementById("api-status");
    const downloadBtn = document.getElementById("download-audio");

    let timerInterval = null;
    let progressInterval = null;
    let currentObjectUrl = null;
    let currentPreviewUrl = null;

    if (!generateBtn || !textInput || !voiceSelect) {
        console.error("Missing required UI elements");
        return;
    }

    function updateCharCount() {
        if (!charCount) return;
        const length = textInput.value.length;
        charCount.textContent = length.toLocaleString();
        charCount.style.color = length > MAX_TEXT_LENGTH ? "#f87171" : "";
    }

    function showError(msg) {
        if (!errorMsg) return;
        errorMsg.textContent = "❌ " + msg;
        errorMsg.classList.remove("hidden");
    }

    function hideError() {
        if (!errorMsg) return;
        errorMsg.classList.add("hidden");
        errorMsg.textContent = "";
    }

    function startTimer() {
        stopTimer();
        const start = Date.now();

        if (progressTimer) {
            progressTimer.textContent = "00:00";
        }

        timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - start) / 1000);
            const mins = String(Math.floor(elapsed / 60)).padStart(2, "0");
            const secs = String(elapsed % 60).padStart(2, "0");

            if (progressTimer) {
                progressTimer.textContent = `${mins}:${secs}`;
            }
        }, 1000);
    }

    function stopTimer() {
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }
    }

    function stopProgressAnimation() {
        if (progressInterval) {
            clearInterval(progressInterval);
            progressInterval = null;
        }
    }

    function showProgress(mode = "main") {
        if (!progressPanel) return;

        progressPanel.classList.remove("hidden");
        generateBtn.disabled = true;
        if (previewBtn) previewBtn.disabled = true;

        if (btnText) {
            btnText.textContent = mode === "preview" ? "جاري إنشاء المعاينة..." : "جاري التوليد...";
        }

        if (progressFill) progressFill.style.width = "15%";

        if (progressStatus) {
            progressStatus.textContent =
                mode === "preview"
                    ? "جاري إنشاء المعاينة الصوتية..."
                    : "جاري تجهيز الحلقة الصوتية...";
        }

        startTimer();
        stopProgressAnimation();

        let width = 15;

        progressInterval = setInterval(() => {
            if (!progressPanel || progressPanel.classList.contains("hidden")) {
                stopProgressAnimation();
                return;
            }

            width = Math.min(width + (mode === "preview" ? 12 : 7), 92);
            if (progressFill) progressFill.style.width = `${width}%`;
        }, mode === "preview" ? 350 : 500);
    }

    function hideProgress(success = false) {
        if (!progressPanel) return;

        stopProgressAnimation();

        if (success) {
            if (progressFill) progressFill.style.width = "100%";
            if (progressStatus) progressStatus.textContent = "تم بنجاح ✨";

            setTimeout(() => {
                progressPanel.classList.add("hidden");
            }, 700);
        } else {
            progressPanel.classList.add("hidden");
        }

        generateBtn.disabled = false;
        if (previewBtn) previewBtn.disabled = false;
        if (btnText) btnText.textContent = "تشغيل الصوت";
        stopTimer();
    }

    function clearOldObjectUrls() {
        if (currentObjectUrl) {
            URL.revokeObjectURL(currentObjectUrl);
            currentObjectUrl = null;
        }

        if (currentPreviewUrl) {
            URL.revokeObjectURL(currentPreviewUrl);
            currentPreviewUrl = null;
        }
    }

    function updateDownloadLink(url, filename = "podcast_episode.wav") {
        if (!downloadBtn) return;

        if (url) {
            downloadBtn.href = url;
            downloadBtn.setAttribute("download", filename);
            downloadBtn.classList.remove("hidden");
        } else {
            downloadBtn.removeAttribute("href");
            downloadBtn.classList.add("hidden");
        }
    }

    function buildEpisodeTitle(prefix = "episode") {
        const now = new Date();
        const pad = (n) => String(n).padStart(2, "0");

        return `${prefix}_${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
    }

    async function checkApi() {
        try {
            const res = await fetch("/api/v1/health");
            const data = await res.json();

            if (apiStatus) {
                apiStatus.textContent = data.status || "ok";
                apiStatus.style.color = "";
            }
        } catch (e) {
            if (apiStatus) {
                apiStatus.textContent = "غير متصل";
                apiStatus.style.color = "#f87171";
            }
        }
    }

    async function loadVoices() {
        try {
            if (voiceStatus) {
                voiceStatus.textContent = "جاري تحميل الأصوات...";
            }

            const res = await fetch("/api/v1/voices");
            const data = await res.json();

            voiceSelect.innerHTML = "";

            const validVoices = (data.voices || []).filter((v) => v.exists);

            if (validVoices.length > 0) {
                validVoices.forEach((v) => {
                    const opt = document.createElement("option");
                    opt.value = v.id;
                    opt.textContent = v.label || v.id;
                    voiceSelect.appendChild(opt);
                });

                if (data.default_voice?.id) {
                    voiceSelect.value = data.default_voice.id;
                }

                if (voiceStatus) {
                    voiceStatus.textContent = `تم تحميل ${validVoices.length} صوت`;
                }
            } else {
                voiceSelect.innerHTML = "<option value=''>لا توجد أصوات متاحة</option>";
                if (voiceStatus) {
                    voiceStatus.textContent = "لا توجد أصوات جاهزة";
                }
            }
        } catch (e) {
            console.error("Failed to load voices", e);
            voiceSelect.innerHTML = "<option value=''>خطأ في تحميل الأصوات</option>";
            if (voiceStatus) {
                voiceStatus.textContent = "فشل تحميل الأصوات";
            }
        }
    }

    async function requestAudioBlob(text, voiceId, { preview = false } = {}) {
        const endpoint = preview ? "/api/v1/speak" : "/api/v1/podcast";

        const payload = preview
            ? {
                  text: text.slice(0, PREVIEW_TEXT_LIMIT),
                  voice_id: voiceId,
                  preview: true,
              }
            : {
                  text: text,
                  voice_id: voiceId,
                  episode_title: buildEpisodeTitle("episode"),
                  bgm_id: "echowave",
                  intro_lead_ms: 2000,
                  silence_between_segments_ms: 500,
                  silence_between_paragraphs_ms: 1400,
                  fast_mode: false,
              };

        const res = await fetch(endpoint, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "audio/wav",
            },
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            let errorMessage = preview
                ? "فشل إنشاء المعاينة الصوتية"
                : "فشل توليد الحلقة الصوتية";

            try {
                const errorData = await res.json();
                if (errorData?.message) {
                    errorMessage = errorData.message;
                }
            } catch (_) {
                // ignore json parse failure
            }

            throw new Error(errorMessage);
        }

        const blob = await res.blob();

        return {
            blob,
            filename: preview ? "preview.wav" : `${payload.episode_title}.wav`,
        };
    }

    async function generateAudio({ autoplay = true, preview = false } = {}) {
        const text = textInput.value.trim();
        const voiceId = voiceSelect.value;

        if (!text) {
            showError("الرجاء إدخال النص أولًا");
            return;
        }

        if (!voiceId) {
            showError("الرجاء اختيار صوت صالح");
            return;
        }

        if (text.length > MAX_TEXT_LENGTH) {
            showError(`النص طويل جدًا، الحد الأقصى هو ${MAX_TEXT_LENGTH} حرف`);
            return;
        }

        hideError();
        showProgress(preview ? "preview" : "main");

        try {
            const { blob, filename } = await requestAudioBlob(text, voiceId, { preview });

            if (preview) {
                if (currentPreviewUrl) {
                    URL.revokeObjectURL(currentPreviewUrl);
                }

                currentPreviewUrl = URL.createObjectURL(blob);

                if (previewPlayer) {
                    previewPlayer.src = currentPreviewUrl;
                    previewPlayer.load();

                    try {
                        await previewPlayer.play();
                    } catch (err) {
                        console.warn("Preview autoplay blocked:", err);
                    }
                }
            } else {
                if (currentObjectUrl) {
                    URL.revokeObjectURL(currentObjectUrl);
                }

                currentObjectUrl = URL.createObjectURL(blob);

                if (normalPlayer) {
                    normalPlayer.src = currentObjectUrl;
                    normalPlayer.load();
                }

                updateDownloadLink(currentObjectUrl, filename);

                if (resultSection) {
                    resultSection.classList.remove("hidden");
                }

                if (autoplay && normalPlayer) {
                    try {
                        await normalPlayer.play();
                    } catch (err) {
                        console.warn("Autoplay blocked:", err);
                    }
                }
            }

            hideProgress(true);
        } catch (err) {
            console.error(err);
            hideProgress(false);
            showError(err.message || "حدث خطأ أثناء توليد الصوت");
        }
    }

    generateBtn.addEventListener("click", async () => {
        await generateAudio({ autoplay: true, preview: false });
    });

    if (previewBtn) {
        previewBtn.addEventListener("click", async () => {
            await generateAudio({ autoplay: true, preview: true });
        });
    }

    if (refreshVoicesBtn) {
        refreshVoicesBtn.addEventListener("click", async () => {
            await loadVoices();
        });
    }

    textInput.addEventListener("input", updateCharCount);

    window.addEventListener("beforeunload", () => {
        clearOldObjectUrls();
    });

    updateDownloadLink(null);
    checkApi();
    loadVoices();
    updateCharCount();
});