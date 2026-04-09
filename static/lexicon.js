document.addEventListener("DOMContentLoaded", () => {
    const originalInput = document.getElementById("original-input");
    const formattedInput = document.getElementById("formatted-input");
    const categorySelect = document.getElementById("category-select");
    const previewWordBtn = document.getElementById("preview-word-btn");
    const addWordBtn = document.getElementById("add-word-btn");
    const refreshLexiconBtn = document.getElementById("refresh-lexicon-btn");
    const manualPreviewPlayer = document.getElementById("manual-preview-player");
    const formMessage = document.getElementById("form-message");
    const lexiconSections = document.getElementById("lexicon-sections");
    const learningList = document.getElementById("learning-list");
    const totalCount = document.getElementById("total-count");
    const lastRefresh = document.getElementById("last-refresh");
    const filterChips = Array.from(document.querySelectorAll(".filter-chip"));

    // Edit modal
    const editLexiconModal = document.getElementById("edit-lexicon-modal");
    const editOriginalInput = document.getElementById("edit-original-input");
    const editFormattedInput = document.getElementById("edit-formatted-input");
    const editCategoryInput = document.getElementById("edit-category-input");
    const editPreviewBtn = document.getElementById("edit-preview-btn");
    const saveEditBtn = document.getElementById("save-edit-btn");
    const cancelEditBtn = document.getElementById("cancel-edit-btn");
    const editPreviewPlayer = document.getElementById("edit-preview-player");
    const editFormMessage = document.getElementById("edit-form-message");

    // Accept learning modal
    const acceptLearningModal = document.getElementById("accept-learning-modal");
    const acceptOriginalInput = document.getElementById("accept-original-input");
    const acceptFormattedInput = document.getElementById("accept-formatted-input");
    const acceptCategorySelect = document.getElementById("accept-category-select");
    const confirmAcceptBtn = document.getElementById("confirm-accept-btn");
    const cancelAcceptBtn = document.getElementById("cancel-accept-btn");
    const acceptFormMessage = document.getElementById("accept-form-message");

    const CATEGORY_LABELS = {
        misc_pronunciation: "كلمات عامة",
        names_pronunciation: "أسماء ومصطلحات",
        tribes_pronunciation: "قبائل",
    };

    let currentEditItem = null;
    let currentLearningItem = null;
    let currentFilter = "all";
    let currentLexicon = {
        misc_pronunciation: [],
        names_pronunciation: [],
        tribes_pronunciation: [],
    };

    function showMessage(text, type = "success", target = formMessage) {
        if (!target) return;
        target.textContent = text;
        target.classList.remove("hidden", "success", "error");
        target.classList.add(type);
    }

    function hideMessage(target = formMessage) {
        if (!target) return;
        target.classList.add("hidden");
        target.textContent = "";
        target.classList.remove("success", "error");
    }

    function resetAudioPlayer(player) {
        if (!player) return;
        try {
            player.pause();
        } catch (_) {}
        player.classList.add("hidden");
        player.removeAttribute("src");
        try {
            player.load();
        } catch (_) {}
    }

    function formatNow() {
        return new Date().toLocaleTimeString("ar");
    }

    function normalizeText(text) {
        return (text || "").trim().replace(/\s+/g, " ");
    }

    function buildPreviewUrl(text) {
        const params = new URLSearchParams({
            text,
            voice: "salem_podcast",
        });
        return `/api/v1/lexicon/preview?${params.toString()}`;
    }

    async function playPreview(text, player = null) {
        const cleanText = normalizeText(text);
        if (!cleanText) {
            throw new Error("النص فارغ");
        }

        const url = buildPreviewUrl(cleanText);

        if (player) {
            player.src = url;
            player.classList.remove("hidden");
            player.load();
            await player.play();
        } else {
            const audio = new Audio(url);
            await audio.play();
        }
    }

    function normalizeLexicon(lexicon) {
        const clean = {
            misc_pronunciation: [],
            names_pronunciation: [],
            tribes_pronunciation: [],
        };

        Object.keys(CATEGORY_LABELS).forEach((cat) => {
            const items = Array.isArray(lexicon?.[cat]) ? lexicon[cat] : [];

            items.forEach((item) => {
                if (Array.isArray(item)) {
                    item.forEach((sub) => {
                        if (sub?.original && sub?.formatted) {
                            clean[cat].push(sub);
                        }
                    });
                } else if (item?.original && item?.formatted) {
                    clean[cat].push(item);
                }
            });
        });

        return clean;
    }

    function openEditLexiconModal(item, category) {
        currentEditItem = {
            original: item.original || "",
            formatted: item.formatted || "",
            category: category || "misc_pronunciation",
        };

        if (editOriginalInput) editOriginalInput.value = currentEditItem.original;
        if (editFormattedInput) editFormattedInput.value = currentEditItem.formatted;
        if (editCategoryInput) {
            editCategoryInput.value =
                CATEGORY_LABELS[currentEditItem.category] || currentEditItem.category;
        }

        hideMessage(editFormMessage);
        resetAudioPlayer(editPreviewPlayer);

        if (editLexiconModal) {
            editLexiconModal.classList.remove("hidden");
            editLexiconModal.setAttribute("aria-hidden", "false");
        }

        setTimeout(() => {
            if (editFormattedInput) {
                editFormattedInput.focus();
                editFormattedInput.select();
            }
        }, 30);
    }

    function closeEditLexiconModal() {
        currentEditItem = null;

        if (editLexiconModal) {
            editLexiconModal.classList.add("hidden");
            editLexiconModal.setAttribute("aria-hidden", "true");
        }

        hideMessage(editFormMessage);
        resetAudioPlayer(editPreviewPlayer);
    }

    function openAcceptLearningModal(change) {
        currentLearningItem = {
            original: change.original || "",
            formatted: change.formatted || "",
            category: "misc_pronunciation",
        };

        if (acceptOriginalInput) acceptOriginalInput.value = currentLearningItem.original;
        if (acceptFormattedInput) acceptFormattedInput.value = currentLearningItem.formatted;
        if (acceptCategorySelect) acceptCategorySelect.value = currentLearningItem.category;

        hideMessage(acceptFormMessage);

        if (acceptLearningModal) {
            acceptLearningModal.classList.remove("hidden");
            acceptLearningModal.setAttribute("aria-hidden", "false");
        }
    }

    function closeAcceptLearningModal() {
        currentLearningItem = null;

        if (acceptLearningModal) {
            acceptLearningModal.classList.add("hidden");
            acceptLearningModal.setAttribute("aria-hidden", "true");
        }

        hideMessage(acceptFormMessage);
    }

    function createItemRow(item, category) {
        const row = document.createElement("div");
        row.className = "lexicon-item";

        const originalBox = document.createElement("div");
        originalBox.className = "word-box";
        originalBox.innerHTML = `
            <span>الأصل</span>
            <strong>${item.original || ""}</strong>
        `;

        const formattedBox = document.createElement("div");
        formattedBox.className = "word-box";
        formattedBox.innerHTML = `
            <span>النطق المحسن</span>
            <strong>${item.formatted || ""}</strong>
        `;

        const actions = document.createElement("div");
        actions.className = "item-actions";

        const playBtn = document.createElement("button");
        playBtn.className = "secondary-btn";
        playBtn.textContent = "▶ تشغيل";
        playBtn.addEventListener("click", async () => {
            try {
                await playPreview(item.formatted || item.original);
            } catch (err) {
                console.error(err);
                alert("فشل تشغيل الصوت");
            }
        });

        const editBtn = document.createElement("button");
        editBtn.className = "secondary-btn";
        editBtn.textContent = "تعديل";
        editBtn.addEventListener("click", () => {
            openEditLexiconModal(item, category);
        });

        const deleteBtn = document.createElement("button");
        deleteBtn.className = "danger-btn";
        deleteBtn.textContent = "حذف";
        deleteBtn.addEventListener("click", async () => {
            const confirmed = confirm(`حذف "${item.original}" من القاموس؟`);
            if (!confirmed) return;

            try {
                const params = new URLSearchParams({
                    original: item.original,
                    category,
                });

                const res = await fetch(`/api/v1/lexicon/delete?${params.toString()}`, {
                    method: "POST",
                });

                const data = await safeJson(res);

                if (!res.ok) {
                    throw new Error(data.detail || "فشل حذف الكلمة");
                }

                await loadLexicon();
            } catch (err) {
                console.error(err);
                alert(err.message || "تعذر حذف الكلمة");
            }
        });

        actions.appendChild(playBtn);
        actions.appendChild(editBtn);
        actions.appendChild(deleteBtn);

        row.appendChild(originalBox);
        row.appendChild(formattedBox);
        row.appendChild(actions);

        return row;
    }

    function renderLexicon(lexicon) {
        if (!lexiconSections) return;

        lexiconSections.innerHTML = "";

        let total = 0;
        Object.keys(CATEGORY_LABELS).forEach((category) => {
            total += Array.isArray(lexicon[category]) ? lexicon[category].length : 0;
        });

        const categoriesToRender =
            currentFilter === "all" ? Object.keys(CATEGORY_LABELS) : [currentFilter];

        categoriesToRender.forEach((category) => {
            const items = Array.isArray(lexicon[category]) ? lexicon[category] : [];

            const group = document.createElement("section");
            group.className = "lexicon-group";

            const header = document.createElement("div");
            header.className = "lexicon-group-header";
            header.innerHTML = `<h3>${CATEGORY_LABELS[category]}</h3>`;

            const body = document.createElement("div");
            body.className = "lexicon-items";

            if (!items.length) {
                const empty = document.createElement("div");
                empty.className = "empty-state";
                empty.textContent = "لا توجد كلمات في هذا القسم";
                body.appendChild(empty);
            } else {
                items.forEach((item) => {
                    body.appendChild(createItemRow(item, category));
                });
            }

            group.appendChild(header);
            group.appendChild(body);
            lexiconSections.appendChild(group);
        });

        if (totalCount) totalCount.textContent = total.toLocaleString("ar");
        if (lastRefresh) lastRefresh.textContent = formatNow();
    }

    function setActiveFilter(filter) {
        currentFilter = filter;

        filterChips.forEach((chip) => {
            chip.classList.toggle("active", chip.dataset.filter === filter);
        });

        renderLexicon(currentLexicon);
    }

    function createLearningRow(change) {
        const row = document.createElement("div");
        row.className = "lexicon-item";

        const originalBox = document.createElement("div");
        originalBox.className = "word-box";
        originalBox.innerHTML = `
            <span>قبل التعديل</span>
            <strong>${change.original || ""}</strong>
        `;

        const formattedBox = document.createElement("div");
        formattedBox.className = "word-box";
        formattedBox.innerHTML = `
            <span>بعد التعديل</span>
            <strong>${change.formatted || ""}</strong>
        `;

        const metaBox = document.createElement("div");
        metaBox.className = "word-box";
        metaBox.innerHTML = `
            <span>النوع / الثقة</span>
            <strong>${change.type || change.change_type || "—"}${change.confidence ? ` • ${Number(change.confidence).toFixed(2)}` : ""}</strong>
        `;

        const actions = document.createElement("div");
        actions.className = "item-actions";

        const acceptBtn = document.createElement("button");
        acceptBtn.className = "primary-btn";
        acceptBtn.textContent = "اعتماد";
        acceptBtn.addEventListener("click", () => {
            openAcceptLearningModal(change);
        });

        const rejectBtn = document.createElement("button");
        rejectBtn.className = "danger-btn";
        rejectBtn.textContent = "رفض";
        rejectBtn.addEventListener("click", async () => {
            try {
                const res = await fetch("/api/v1/learning/reject", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        original: change.original,
                        formatted: change.formatted,
                    }),
                });

                const data = await safeJson(res);

                if (!res.ok) {
                    throw new Error(data.detail || "فشل رفض الكلمة");
                }

                await loadLearningLog();
            } catch (err) {
                console.error(err);
                alert(err.message || "تعذر رفض الكلمة");
            }
        });

        actions.appendChild(acceptBtn);
        actions.appendChild(rejectBtn);

        row.appendChild(originalBox);
        row.appendChild(formattedBox);
        row.appendChild(metaBox);
        row.appendChild(actions);

        return row;
    }

    function renderLearning(items) {
        if (!learningList) return;

        learningList.innerHTML = "";

        const allChanges = [];

        items.forEach((entry) => {
            const changes = Array.isArray(entry.changes)
                ? entry.changes
                : Array.isArray(entry.learned)
                    ? entry.learned
                    : [];

            changes.forEach((change) => {
                if (change && typeof change === "object") {
                    allChanges.push(change);
                }
            });
        });

        if (!allChanges.length) {
            learningList.innerHTML = `<div class="empty-state">لا توجد بيانات تعلم حالياً</div>`;
            return;
        }

        allChanges.slice(0, 20).forEach((change) => {
            learningList.appendChild(createLearningRow(change));
        });
    }

    async function safeJson(res) {
        try {
            return await res.json();
        } catch (_) {
            return {};
        }
    }

    async function loadLearningLog() {
        if (!learningList) return;

        try {
            const res = await fetch("/api/v1/learning-log");
            const data = await safeJson(res);

            if (!res.ok) {
                throw new Error(data.detail || "فشل تحميل سجل التعلم");
            }

            const items = Array.isArray(data.items) ? data.items : [];
            renderLearning(items);
        } catch (err) {
            console.error(err);
            learningList.innerHTML = `<div class="empty-state">تعذر تحميل سجل التعلم</div>`;
        }
    }

    async function loadLexicon() {
        try {
            const res = await fetch("/api/v1/lexicon");
            const data = await safeJson(res);

            if (!res.ok) {
                throw new Error(data.detail || "فشل تحميل القاموس");
            }

            currentLexicon = normalizeLexicon(data.lexicon || {});
            renderLexicon(currentLexicon);
        } catch (err) {
            console.error(err);
            if (lexiconSections) {
                lexiconSections.innerHTML = `<div class="empty-state">تعذر تحميل القاموس</div>`;
            }
            if (totalCount) totalCount.textContent = "0";
            if (lastRefresh) lastRefresh.textContent = "—";
        }
    }

    previewWordBtn?.addEventListener("click", async () => {
        hideMessage();

        const formatted = normalizeText(formattedInput?.value);
        const original = normalizeText(originalInput?.value);
        const text = formatted || original;

        if (!text) {
            showMessage("اكتب الكلمة أو النطق أولًا", "error");
            return;
        }

        try {
            await playPreview(text, manualPreviewPlayer);
        } catch (err) {
            console.error(err);
            showMessage("فشل تشغيل المعاينة", "error");
        }
    });

    addWordBtn?.addEventListener("click", async () => {
        hideMessage();

        const original = normalizeText(originalInput?.value);
        const formatted = normalizeText(formattedInput?.value);
        const category = categorySelect?.value || "misc_pronunciation";

        if (!original || !formatted) {
            showMessage("أدخل الكلمة الأصلية والنطق المحسن", "error");
            return;
        }

        try {
            const params = new URLSearchParams({
                original,
                formatted,
                category,
            });

            const res = await fetch(`/api/v1/lexicon/add?${params.toString()}`, {
                method: "POST",
            });

            const data = await safeJson(res);

            if (!res.ok) {
                throw new Error(data.detail || "فشل إضافة الكلمة");
            }

            if (originalInput) originalInput.value = "";
            if (formattedInput) formattedInput.value = "";
            resetAudioPlayer(manualPreviewPlayer);

            showMessage("تمت إضافة الكلمة للقاموس", "success");

            await loadLexicon();
        } catch (err) {
            console.error(err);
            showMessage(err.message || "تعذر إضافة الكلمة", "error");
        }
    });

    refreshLexiconBtn?.addEventListener("click", async () => {
        await loadLexicon();
        await loadLearningLog();
        hideMessage();
    });

    filterChips.forEach((chip) => {
        chip.addEventListener("click", () => {
            setActiveFilter(chip.dataset.filter || "all");
        });
    });

    editPreviewBtn?.addEventListener("click", async () => {
        hideMessage(editFormMessage);

        const original = normalizeText(editOriginalInput?.value);
        const formatted = normalizeText(editFormattedInput?.value);
        const text = formatted || original;

        if (!text) {
            showMessage("اكتب النطق المحسن أولًا", "error", editFormMessage);
            return;
        }

        try {
            await playPreview(text, editPreviewPlayer);
        } catch (err) {
            console.error(err);
            showMessage("فشل تشغيل المعاينة", "error", editFormMessage);
        }
    });

    saveEditBtn?.addEventListener("click", async () => {
        hideMessage(editFormMessage);

        if (!currentEditItem) {
            showMessage("لا توجد كلمة محددة للتعديل", "error", editFormMessage);
            return;
        }

        const original = normalizeText(editOriginalInput?.value);
        const formatted = normalizeText(editFormattedInput?.value);
        const category = currentEditItem.category || "misc_pronunciation";

        if (!original || !formatted) {
            showMessage("أدخل النطق المحسن", "error", editFormMessage);
            return;
        }

        try {
            const params = new URLSearchParams({
                original,
                formatted,
                category,
            });

            const updateRes = await fetch(`/api/v1/lexicon/update?${params.toString()}`, {
                method: "POST",
            });

            const updateData = await safeJson(updateRes);

            if (!updateRes.ok) {
                throw new Error(updateData.detail || "فشل تعديل الكلمة");
            }

            const learnRes = await fetch("/api/v1/lexicon/learn", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    original: currentEditItem.original,
                    edited: formatted,
                    category,
                }),
            });

            if (!learnRes.ok) {
                console.warn("تعذر تنفيذ التعلم التلقائي");
            }

            showMessage("تم حفظ التعديل بنجاح", "success", editFormMessage);

            await loadLexicon();
            await loadLearningLog();

            setTimeout(() => {
                closeEditLexiconModal();
            }, 500);
        } catch (err) {
            console.error(err);
            showMessage(err.message || "تعذر حفظ التعديل", "error", editFormMessage);
        }
    });

    confirmAcceptBtn?.addEventListener("click", async () => {
        hideMessage(acceptFormMessage);

        if (!currentLearningItem) {
            showMessage("لا توجد كلمة محددة للاعتماد", "error", acceptFormMessage);
            return;
        }

        const original = normalizeText(acceptOriginalInput?.value);
        const formatted = normalizeText(acceptFormattedInput?.value);
        const category = acceptCategorySelect?.value || "misc_pronunciation";

        if (!original || !formatted) {
            showMessage("بيانات الاعتماد غير مكتملة", "error", acceptFormMessage);
            return;
        }

        try {
            const res = await fetch("/api/v1/learning/accept", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    original,
                    formatted,
                    category,
                }),
            });

            const data = await safeJson(res);

            if (!res.ok) {
                throw new Error(data.detail || "فشل اعتماد الكلمة");
            }

            showMessage("تم اعتماد الكلمة وإضافتها للقاموس", "success", acceptFormMessage);

            await loadLexicon();
            await loadLearningLog();

            setTimeout(() => {
                closeAcceptLearningModal();
            }, 400);
        } catch (err) {
            console.error(err);
            showMessage(err.message || "تعذر اعتماد الكلمة", "error", acceptFormMessage);
        }
    });

    cancelAcceptBtn?.addEventListener("click", () => {
        closeAcceptLearningModal();
    });

    cancelEditBtn?.addEventListener("click", () => {
        closeEditLexiconModal();
    });

    editLexiconModal?.addEventListener("click", (event) => {
        const target = event.target;
        if (target instanceof HTMLElement && target.dataset.closeEditModal === "true") {
            closeEditLexiconModal();
        }
    });

    acceptLearningModal?.addEventListener("click", (event) => {
        const target = event.target;
        if (target instanceof HTMLElement && target.dataset.closeAcceptModal === "true") {
            closeAcceptLearningModal();
        }
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            if (editLexiconModal && !editLexiconModal.classList.contains("hidden")) {
                closeEditLexiconModal();
            }
            if (acceptLearningModal && !acceptLearningModal.classList.contains("hidden")) {
                closeAcceptLearningModal();
            }
        }
    });

    loadLexicon();
    loadLearningLog();
});