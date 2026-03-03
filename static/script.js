document.addEventListener("DOMContentLoaded", () => {
    // Generate Elements
    const generateBtn = document.getElementById("generate-btn");
    const textInput = document.getElementById("text-input");
    const voiceSelect = document.getElementById("voice-select");
    const bgmSelect = document.getElementById("bgm-select");
    const spinner = document.getElementById("loading-spinner");
    const btnText = document.querySelector(".btn-text");
    const previewBtn = document.getElementById("preview-btn");
    const previewPlayer = document.getElementById("preview-player");

    // Result Elements
    const resultSection = document.getElementById("result-section");
    const normalPlayer = document.getElementById("normal-player");
    const cinematicPlayer = document.getElementById("cinematic-player");
    const normalDownload = document.getElementById("normal-download");
    const cinematicDownload = document.getElementById("cinematic-download");
    const sessionIdInput = document.getElementById("current-session-id");
    const segmentsList = document.getElementById("segments-list");
    const bgmPreviewBtn = document.getElementById("bgm-preview-btn");
    const bgmPreviewPlayer = document.getElementById("bgm-preview-player");

    // Lexicon Elements
    const lexOrig = document.getElementById("lex-orig");
    const lexCorr = document.getElementById("lex-corr");
    const addLexiconBtn = document.getElementById("add-lexicon-btn");
    const lexiconMsg = document.getElementById("lexicon-msg");

    const errorMsg = document.getElementById("error-message");

    // --- 1. Add Lexicon Entry ---
    addLexiconBtn.addEventListener("click", async () => {
        const orig = lexOrig.value.trim();
        const corr = lexCorr.value.trim();
        if (!orig || !corr) return;

        try {
            const res = await fetch("/api/v1/lexicon", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ original: orig, corrected: corr })
            });
            const data = await res.json();
            if (res.ok) {
                lexiconMsg.textContent = "✅ " + data.message;
                setTimeout(() => lexiconMsg.textContent = "", 3000);
                lexOrig.value = ""; lexCorr.value = "";
            }
        } catch (e) {
            console.error("Lexicon add failed", e);
        }
    });

    // --- 1.5 Fetch Voices ---
    async function loadVoices() {
        try {
            const res = await fetch("/api/v1/voices");
            const data = await res.json();
            voiceSelect.innerHTML = "";
            if (data.voices && data.voices.length > 0) {
                data.voices.forEach(v => {
                    const opt = document.createElement("option");
                    opt.value = v.id;
                    opt.textContent = v.name;
                    voiceSelect.appendChild(opt);
                });
            } else {
                voiceSelect.innerHTML = "<option value=''>لا توجد أصوات متاحة</option>";
            }
        } catch (e) {
            console.error("Failed to load voices", e);
            voiceSelect.innerHTML = "<option value=''>خطأ في تحميل الأصوات</option>";
        }
    }
    loadVoices();

    // Preview Voice Functionality
    previewBtn.addEventListener("click", () => {
        const selectedVoice = voiceSelect.value;
        if (!selectedVoice) return;

        // Stop current if playing
        previewPlayer.pause();
        previewPlayer.src = `/api/v1/voices/${selectedVoice}/preview`;

        // Quick visual feedback
        previewBtn.textContent = "⏳";

        previewPlayer.play().then(() => {
            previewBtn.textContent = "🔊";
        }).catch(err => {
            console.error("Preview playback failed", err);
            previewBtn.textContent = "❌";
        });
    });

    // Reset button icon when preview ends
    previewPlayer.addEventListener("ended", () => {
        previewBtn.textContent = "▶️";
    });

    // --- 1.6 Fetch BGMs ---
    async function loadBgms() {
        try {
            const res = await fetch("/api/v1/bgms");
            const data = await res.json();
            bgmSelect.innerHTML = "<option value=''>بدون مسار محدد (تلقائي)</option>";
            if (data.bgms && data.bgms.length > 0) {
                data.bgms.forEach(bgm => {
                    const opt = document.createElement("option");
                    opt.value = bgm.id;
                    opt.textContent = bgm.name;
                    bgmSelect.appendChild(opt);
                });
            }
        } catch (e) {
            console.error("Failed to load BGMs", e);
        }
    }
    loadBgms();

    // Preview BGM Functionality
    bgmPreviewBtn.addEventListener("click", () => {
        const selectedBgm = bgmSelect.value;
        if (!selectedBgm) return;

        bgmPreviewPlayer.pause();
        bgmPreviewPlayer.src = `/api/v1/bgms/${selectedBgm}/preview`;
        bgmPreviewBtn.textContent = "⏳";

        bgmPreviewPlayer.play().then(() => {
            bgmPreviewBtn.textContent = "🔊";
        }).catch(err => {
            console.error("BGM preview playback failed", err);
            bgmPreviewBtn.textContent = "❌";
        });
    });

    bgmPreviewPlayer.addEventListener("ended", () => {
        bgmPreviewBtn.textContent = "▶️";
    });

    // --- 2. Main Generation ---
    generateBtn.addEventListener("click", () => startProcess("/api/v1/generate", {
        text: textInput.value.trim(),
        voice_id: voiceSelect.value,
        bgm_id: bgmSelect.value
    }));

    async function startProcess(url, bodyData) {
        if (!bodyData.text && !bodyData.new_text) {
            showError("الرجاء إدخال النص أولاً!");
            return;
        }

        setLoading(true);
        hideError();

        // Only hide results if it's a completely new generation (not a heal)
        if (url === "/api/v1/generate") {
            resultSection.classList.add("hidden");
        }

        try {
            const response = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(bodyData)
            });

            const data = await response.json();

            if (!response.ok) throw new Error(data.detail || "خطأ غير متوقع من الخادم");

            pollStatus(data.task_id);
        } catch (err) {
            showError(err.message);
            setLoading(false);
        }
    }

    async function pollStatus(taskId) {
        try {
            const statusRes = await fetch(`/api/v1/status/${taskId}`);
            const statusData = await statusRes.json();

            if (!statusRes.ok) {
                throw new Error(statusData.detail || "حدث خطأ غير متوقع أثناء معالجة الصوت خلف الكواليس.");
            }

            if (statusData.status === "completed") {
                populateResults(statusData);
                setLoading(false);
            } else if (statusData.status === "error") {
                throw new Error(statusData.message || "فشل توليد الصوت");
            } else {
                setTimeout(() => pollStatus(taskId), 2000);
            }
        } catch (err) {
            showError(err.message);
            setLoading(false);
        }
    }

    // --- 3. Render Results & Self-Healing ---
    function populateResults(data) {
        const ts = new Date().getTime();

        // 1. Audio Players
        normalPlayer.src = data.normal_url + `?t=${ts}`;
        cinematicPlayer.src = data.cinematic_url + `?t=${ts}`;

        normalDownload.href = data.normal_url + "?download=1";
        cinematicDownload.href = data.cinematic_url + "?download=1";

        if (data.session_id) sessionIdInput.value = data.session_id;

        // 2. Render Segments for Self-Healing
        if (data.segments) {
            segmentsList.innerHTML = "";
            data.segments.forEach((seg, index) => {
                const segDiv = document.createElement("div");
                segDiv.style.cssText = "display: flex; gap: 10px; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px; align-items:center;";

                const segInput = document.createElement("input");
                segInput.type = "text";
                segInput.value = seg.text;
                segInput.style.cssText = "flex: 1; padding: 8px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.1); background: transparent; color: white;";
                segInput.id = `seg-input-${index}`;

                const healBtn = document.createElement("button");
                healBtn.className = "secondary-btn";
                healBtn.innerHTML = "🔄 تصحيح";
                healBtn.style.padding = "8px 15px";
                healBtn.onclick = () => healSegment(index, segInput.value);

                segDiv.appendChild(segInput);
                segDiv.appendChild(healBtn);
                segmentsList.appendChild(segDiv);
            });
        }

        resultSection.classList.remove("hidden");
    }

    function healSegment(index, newText) {
        setLoading(true);
        startProcess("/api/v1/session/regenerate", {
            session_id: sessionIdInput.value,
            segment_index: index,
            new_text: newText
        });
    }

    // --- UI Helpers ---
    function setLoading(isLoading) {
        if (isLoading) {
            generateBtn.disabled = true;
            spinner.classList.remove("hidden");
            btnText.textContent = "جاري المعالجة...";
        } else {
            generateBtn.disabled = false;
            spinner.classList.add("hidden");
            btnText.textContent = "توليد (عادي + سينمائي)";
        }
    }

    function showError(msg) {
        errorMsg.textContent = "❌ " + msg;
        errorMsg.classList.remove("hidden");
    }

    function hideError() {
        errorMsg.classList.add("hidden");
    }
});
