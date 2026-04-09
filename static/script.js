document.addEventListener("DOMContentLoaded", () => {
    const MAX_TEXT_LENGTH = 2000;

    const textInput = document.getElementById("text-input");
    const charCount = document.getElementById("char-count");
    const generateBtn = document.getElementById("generate-btn");
    const voiceSelect = document.getElementById("voice-select");
    const refreshVoicesBtn = document.getElementById("refresh-voices-btn");
    const voiceStatus = document.getElementById("voice-status");
    const apiStatus = document.getElementById("api-status");
    const errorMsg = document.getElementById("error-message");
    const segmentsList = document.getElementById("segments-list");
    const progressPanel = document.getElementById("progress-panel");
    const progressFill = document.getElementById("progress-fill");
    const progressStatus = document.getElementById("progress-status");
    const progressTimer = document.getElementById("progress-timer");

    let timerInterval = null;
    let progressInterval = null;

    function showError(msg) {
        if (!errorMsg) return;
        errorMsg.textContent = "❌ " + msg;
        errorMsg.classList.remove("hidden");
    }

    function hideError() {
        if (!errorMsg) return;
        errorMsg.textContent = "";
        errorMsg.classList.add("hidden");
    }

    function updateCharCount() {
        if (!textInput || !charCount) return;
        const length = textInput.value.length;
        charCount.textContent = length.toLocaleString();
        charCount.style.color = length > MAX_TEXT_LENGTH ? "#f87171" : "";
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

    function showProgress(message = "جاري إنشاء الصوت...") {
        if (!progressPanel) return;

        progressPanel.classList.remove("hidden");
        if (progressStatus) progressStatus.textContent = message;
        if (progressFill) progressFill.style.width = "15%";
        if (generateBtn) generateBtn.disabled = true;

        startTimer();

        let width = 15;
        if (progressInterval) clearInterval(progressInterval);

        progressInterval = setInterval(() => {
            width = Math.min(width + 8, 90);
            if (progressFill) {
                progressFill.style.width = `${width}%`;
            }
        }, 400);
    }

    function hideProgress(success = false) {
        if (progressInterval) {
            clearInterval(progressInterval);
            progressInterval = null;
        }

        if (!progressPanel) return;

        if (success) {
            if (progressFill) progressFill.style.width = "100%";
            if (progressStatus) progressStatus.textContent = "تم بنجاح ✨";

            setTimeout(() => {
                progressPanel.classList.add("hidden");
            }, 600);
        } else {
            progressPanel.classList.add("hidden");
        }

        if (generateBtn) generateBtn.disabled = false;
        stopTimer();
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
            if (voiceStatus) voiceStatus.textContent = "جاري تحميل الأصوات...";

            const res = await fetch("/api/v1/voices");
            const data = await res.json();

            if (!voiceSelect) return;
            voiceSelect.innerHTML = "";

            const validVoices = (data.voices || []).filter((v) => v.exists);

            if (!validVoices.length) {
                voiceSelect.innerHTML = "<option value=''>لا توجد أصوات</option>";
                if (voiceStatus) voiceStatus.textContent = "لا توجد أصوات جاهزة";
                return;
            }

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
        } catch (e) {
            if (voiceSelect) {
                voiceSelect.innerHTML = "<option value=''>خطأ في تحميل الأصوات</option>";
            }
            if (voiceStatus) {
                voiceStatus.textContent = "فشل تحميل الأصوات";
            }
        }
    }

    async function fetchSegments() {
        const res = await fetch("/api/v1/segments");
        if (!res.ok) throw new Error("فشل تحميل المقاطع");
        return res.json();
    }

    async function createSegment(text, voiceId) {
        const res = await fetch("/api/v1/segments/create", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, voice_id: voiceId }),
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "فشل إنشاء المقطع");
        return data;
    }

    async function updateSegment(id, text, voiceId) {
        const res = await fetch("/api/v1/segments/update", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id, text, voice_id: voiceId }),
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "فشل تعديل المقطع");
        return data;
    }

    async function deleteSegment(id) {
        const res = await fetch("/api/v1/segments/delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id }),
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "فشل حذف المقطع");
        return data;
    }

    async function mergeSegmentsOnlyRequest(segmentIds) {
        const res = await fetch("/api/v1/merge/segments-only", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                segment_ids: segmentIds,
                silence_ms: 3000,
                output_name: "merged_segments",
                delete_segments_after_merge: true,
            }),
        });

        if (!res.ok) {
            let message = "فشل دمج المقاطع";
            try {
                const data = await res.json();
                message = data.detail || message;
            } catch (_) {}
            throw new Error(message);
        }

        return res.blob();
    }

    async function mergeFullEpisodeRequest(segmentIds) {
        const res = await fetch("/api/v1/merge/full-episode", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                segment_ids: segmentIds,
                include_intro: true,
                include_outro: true,
                include_silence: true,
                silence_ms: 3000,
                output_name: "full_episode",
                delete_segments_after_merge: true,
            }),
        });

        if (!res.ok) {
            let message = "فشل إنشاء الحلقة";
            try {
                const data = await res.json();
                message = data.detail || message;
            } catch (_) {}
            throw new Error(message);
        }

        return res.blob();
    }

    async function mergeWithBgmRequest(segmentIds, bgmId = "echowave") {
        const res = await fetch("/api/v1/merge/full-episode-with-bgm", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                segment_ids: segmentIds,
                include_intro: true,
                include_outro: true,
                include_silence: true,
                silence_ms: 3000,
                bgm_id: bgmId,
                output_name: "full_episode_bgm",
                delete_segments_after_merge: true,
            }),
        });

        if (!res.ok) {
            let message = "فشل إضافة الموسيقى";
            try {
                const data = await res.json();
                message = data.detail || message;
            } catch (_) {}
            throw new Error(message);
        }

        return res.blob();
    }

    function downloadFile(blob, filename) {
        const url = window.URL.createObjectURL(blob);

        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();

        window.URL.revokeObjectURL(url);
    }

    function getSelectedSegments() {
        const checked = document.querySelectorAll(".segment-check:checked");
        return Array.from(checked).map((el) => el.value);
    }

    function closeAllEditBoxes() {
        document.querySelectorAll(".edit-box").forEach((box) => {
            box.classList.add("hidden");
        });
    }

    function buildSegmentCard(segment) {
        const card = document.createElement("div");
        card.className = "audio-card glass-subpanel segment-card";

        const audioUrl = `/api/v1/audio/${segment.filename}?t=${Date.now()}`;

        const headerRow = document.createElement("div");
        headerRow.style.display = "flex";
        headerRow.style.alignItems = "center";
        headerRow.style.justifyContent = "space-between";
        headerRow.style.gap = "12px";
        headerRow.style.marginBottom = "12px";

        const title = document.createElement("h3");
        title.textContent = `🎧 مقطع ${segment.id}`;
        title.style.marginBottom = "0";

        const selectBox = document.createElement("input");
        selectBox.type = "checkbox";
        selectBox.className = "segment-check";
        selectBox.value = segment.id;
        selectBox.title = "تحديد هذا المقطع للدمج";

        headerRow.appendChild(title);
        headerRow.appendChild(selectBox);

        const textView = document.createElement("p");
        textView.className = "segment-text";
        textView.textContent = segment.text;

        const audio = document.createElement("audio");
        audio.controls = true;
        audio.preload = "none";
        audio.src = audioUrl;

        const actionRow = document.createElement("div");
        actionRow.className = "action-row segment-actions";

        const editBtn = document.createElement("button");
        editBtn.type = "button";
        editBtn.className = "secondary-btn";
        editBtn.textContent = "تعديل";

        const deleteBtn = document.createElement("button");
        deleteBtn.type = "button";
        deleteBtn.className = "secondary-btn";
        deleteBtn.textContent = "حذف";

        const downloadLink = document.createElement("a");
        downloadLink.className = "secondary-btn";
        downloadLink.href = audioUrl;
        downloadLink.download = segment.filename;
        downloadLink.textContent = "تحميل";

        actionRow.appendChild(downloadLink);
        actionRow.appendChild(deleteBtn);
        actionRow.appendChild(editBtn);

        const editBox = document.createElement("div");
        editBox.className = "edit-box hidden";

        const editTextarea = document.createElement("textarea");
        editTextarea.className = "main-textarea edit-text";
        editTextarea.maxLength = 2000;
        editTextarea.value = segment.text;

        const editActions = document.createElement("div");
        editActions.className = "action-row";

        const cancelEditBtn = document.createElement("button");
        cancelEditBtn.type = "button";
        cancelEditBtn.className = "secondary-btn";
        cancelEditBtn.textContent = "إلغاء";

        const saveEditBtn = document.createElement("button");
        saveEditBtn.type = "button";
        saveEditBtn.className = "secondary-btn";
        saveEditBtn.textContent = "حفظ التعديل";

        editActions.appendChild(cancelEditBtn);
        editActions.appendChild(saveEditBtn);

        editBox.appendChild(editTextarea);
        editBox.appendChild(editActions);

        editBtn.addEventListener("click", () => {
            closeAllEditBoxes();
            editTextarea.value = segment.text;
            editBox.classList.remove("hidden");
            editTextarea.focus();
        });

        cancelEditBtn.addEventListener("click", () => {
            editBox.classList.add("hidden");
            editTextarea.value = segment.text;
        });

        saveEditBtn.addEventListener("click", async () => {
            const newText = editTextarea.value.trim();
            const voiceId = voiceSelect?.value || "";

            if (!newText) {
                showError("النص المعدل فارغ");
                return;
            }

            if (!voiceId) {
                showError("اختر صوتًا");
                return;
            }

            if (newText.length > MAX_TEXT_LENGTH) {
                showError("النص المعدل يتجاوز 2000 حرف");
                return;
            }

            saveEditBtn.disabled = true;
            saveEditBtn.textContent = "جاري الحفظ...";

            try {
                hideError();
                showProgress("جاري تحديث المقطع...");
                await updateSegment(segment.id, newText, voiceId);
                await renderSegments();
                hideProgress(true);
            } catch (err) {
                console.error("Update segment failed:", err);
                hideProgress(false);
                showError(err.message || "فشل تعديل المقطع");
            } finally {
                saveEditBtn.disabled = false;
                saveEditBtn.textContent = "حفظ التعديل";
            }
        });

        deleteBtn.addEventListener("click", async () => {
            try {
                hideError();
                showProgress("جاري حذف المقطع...");
                await deleteSegment(segment.id);
                await renderSegments();
                hideProgress(true);
            } catch (err) {
                console.error("Delete segment failed:", err);
                hideProgress(false);
                showError(err.message || "فشل حذف المقطع");
            }
        });

        card.appendChild(headerRow);
        card.appendChild(textView);
        card.appendChild(audio);
        card.appendChild(actionRow);
        card.appendChild(editBox);

        return card;
    }

    async function renderSegments() {
        if (!segmentsList) return;

        segmentsList.innerHTML = "";

        const data = await fetchSegments();
        const segments = data.segments || [];

        if (!segments.length) {
            segmentsList.innerHTML = `
                <div class="audio-card glass-subpanel">
                    <h3>لا توجد مقاطع بعد</h3>
                    <p style="color:#94a3b8;">أنشئ أول مقطع ليظهر هنا.</p>
                </div>
            `;
            return;
        }

        segments.forEach((segment) => {
            segmentsList.appendChild(buildSegmentCard(segment));
        });
    }

    async function mergeSegmentsOnly() {
        const ids = getSelectedSegments();

        if (!ids.length) {
            showError("اختر مقاطع أولاً");
            return;
        }

        try {
            hideError();
            showProgress("جاري دمج المقاطع...");
            const blob = await mergeSegmentsOnlyRequest(ids);
            downloadFile(blob, "merged_segments.wav");
            await renderSegments();
            hideProgress(true);
        } catch (err) {
            hideProgress(false);
            showError(err.message || "فشل دمج المقاطع");
        }
    }

    async function mergeFullEpisode() {
        const ids = getSelectedSegments();

        if (!ids.length) {
            showError("اختر مقاطع أولاً");
            return;
        }

        try {
            hideError();
            showProgress("جاري إنشاء الحلقة الكاملة...");
            const blob = await mergeFullEpisodeRequest(ids);
            downloadFile(blob, "full_episode.wav");
            await renderSegments();
            hideProgress(true);
        } catch (err) {
            hideProgress(false);
            showError(err.message || "فشل إنشاء الحلقة");
        }
    }

    async function mergeWithBgm() {
        const ids = getSelectedSegments();

        if (!ids.length) {
            showError("اختر مقاطع أولاً");
            return;
        }

        try {
            hideError();
            showProgress("جاري إنشاء الحلقة مع الموسيقى...");
            const blob = await mergeWithBgmRequest(ids, "echowave");
            downloadFile(blob, "full_episode_bgm.wav");
            await renderSegments();
            hideProgress(true);
        } catch (err) {
            hideProgress(false);
            showError(err.message || "فشل إنشاء الحلقة مع الموسيقى");
        }
    }

    if (generateBtn) {
        generateBtn.addEventListener("click", async () => {
            const text = textInput?.value.trim() || "";
            const voiceId = voiceSelect?.value || "";

            if (!text) {
                showError("اكتب النص أولًا");
                return;
            }

            if (!voiceId) {
                showError("اختر صوتًا");
                return;
            }

            if (text.length > MAX_TEXT_LENGTH) {
                showError("الحد الأقصى 2000 حرف");
                return;
            }

            try {
                hideError();
                showProgress("جاري إنشاء المقطع...");
                await createSegment(text, voiceId);

                if (textInput) {
                    textInput.value = "";
                }

                updateCharCount();
                await renderSegments();
                hideProgress(true);
            } catch (err) {
                hideProgress(false);
                showError(err.message || "حدث خطأ أثناء الإنشاء");
            }
        });
    }

    if (refreshVoicesBtn) {
        refreshVoicesBtn.addEventListener("click", loadVoices);
    }

    if (textInput) {
        textInput.addEventListener("input", updateCharCount);
    }

    // expose merge actions to window for HTML buttons
    window.mergeSegmentsOnly = mergeSegmentsOnly;
    window.mergeFullEpisode = mergeFullEpisode;
    window.mergeWithBgm = mergeWithBgm;

    checkApi();
    loadVoices();
    renderSegments();
    updateCharCount();
});