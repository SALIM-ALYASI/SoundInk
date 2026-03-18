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
    const totalCount = document.getElementById("total-count");
    const lastRefresh = document.getElementById("last-refresh");

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

    const CATEGORY_LABELS = {
        misc_pronunciation: "كلمات عامة",
        names_pronunciation: "أسماء ومصطلحات",
        tribes_pronunciation: "قبائل",
    };

    let currentEditItem = null;

    function showMessage(text, type = "success", target = formMessage) {
        target.textContent = text;
        target.classList.remove("hidden", "success", "error");
        target.classList.add(type);
    }

    function hideMessage(target = formMessage) {
        target.classList.add("hidden");
        target.textContent = "";
        target.classList.remove("success", "error");
    }

    function resetAudioPlayer(player) {
        if (!player) return;
        player.pause();
        player.classList.add("hidden");
        player.removeAttribute("src");
        player.load();
    }

    function formatNow() {
        const now = new Date();
        return now.toLocaleTimeString("ar");
    }

    function buildPreviewUrl(text) {
        const params = new URLSearchParams({
            text,
            voice: "salem_podcast",
        });
        return `/api/v1/lexicon/preview?${params.toString()}`;
    }

    async function playPreview(text, player = null) {
        const cleanText = (text || "").trim();
        if (!cleanText) {
            throw new Error("النص فارغ");
        }

        const targetPlayer = player || new Audio();
        const url = buildPreviewUrl(cleanText);

        if (player) {
            player.src = url;
            player.classList.remove("hidden");
            player.load();
            await player.play();
        } else {
            targetPlayer.src = url;
            await targetPlayer.play();
        }
    }

    function openEditLexiconModal(item, category) {
        currentEditItem = {
            original: item.original || "",
            formatted: item.formatted || "",
            category: category || "misc_pronunciation",
        };

        editOriginalInput.value = currentEditItem.original;
        editFormattedInput.value = currentEditItem.formatted;
        editCategoryInput.value = CATEGORY_LABELS[currentEditItem.category] || currentEditItem.category;

        hideMessage(editFormMessage);
        resetAudioPlayer(editPreviewPlayer);

        editLexiconModal.classList.remove("hidden");
        editLexiconModal.setAttribute("aria-hidden", "false");

        setTimeout(() => {
            editFormattedInput.focus();
            editFormattedInput.select();
        }, 30);
    }

    function closeEditLexiconModal() {
        currentEditItem = null;
        editLexiconModal.classList.add("hidden");
        editLexiconModal.setAttribute("aria-hidden", "true");
        hideMessage(editFormMessage);
        resetAudioPlayer(editPreviewPlayer);
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

                if (!res.ok) {
                    throw new Error("فشل حذف الكلمة");
                }

                await loadLexicon();
            } catch (err) {
                console.error(err);
                alert("تعذر حذف الكلمة");
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
        lexiconSections.innerHTML = "";

        let total = 0;

        Object.keys(CATEGORY_LABELS).forEach((category) => {
            const items = Array.isArray(lexicon[category]) ? lexicon[category] : [];
            total += items.length;

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

        totalCount.textContent = total.toLocaleString("ar");
        lastRefresh.textContent = formatNow();
    }

    async function loadLexicon() {
        try {
            const res = await fetch("/api/v1/lexicon");
            if (!res.ok) {
                throw new Error("فشل تحميل القاموس");
            }

            const data = await res.json();
            renderLexicon(data.lexicon || {});
        } catch (err) {
            console.error(err);
            lexiconSections.innerHTML = `<div class="empty-state">تعذر تحميل القاموس</div>`;
        }
    }

    previewWordBtn.addEventListener("click", async () => {
        hideMessage();

        const formatted = formattedInput.value.trim();
        const original = originalInput.value.trim();
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

    addWordBtn.addEventListener("click", async () => {
        hideMessage();

        const original = originalInput.value.trim();
        const formatted = formattedInput.value.trim();
        const category = categorySelect.value;

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

            if (!res.ok) {
                throw new Error("فشل إضافة الكلمة");
            }

            originalInput.value = "";
            formattedInput.value = "";
            resetAudioPlayer(manualPreviewPlayer);

            showMessage("تمت إضافة الكلمة للقاموس", "success");
            await loadLexicon();
        } catch (err) {
            console.error(err);
            showMessage("تعذر إضافة الكلمة", "error");
        }
    });

    refreshLexiconBtn.addEventListener("click", async () => {
        await loadLexicon();
        hideMessage();
    });

    editPreviewBtn?.addEventListener("click", async () => {
        hideMessage(editFormMessage);

        const original = editOriginalInput.value.trim();
        const formatted = editFormattedInput.value.trim();
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

        const original = editOriginalInput.value.trim();
        const formatted = editFormattedInput.value.trim();
        const category = currentEditItem.category;

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

            const res = await fetch(`/api/v1/lexicon/update?${params.toString()}`, {
                method: "POST",
            });

            let data = {};
            try {
                data = await res.json();
            } catch (_) {
                data = {};
            }

            if (!res.ok) {
                throw new Error(data.detail || "فشل تعديل الكلمة");
            }

            showMessage("تم حفظ التعديل بنجاح", "success", editFormMessage);
            await loadLexicon();

            setTimeout(() => {
                closeEditLexiconModal();
            }, 500);
        } catch (err) {
            console.error(err);
            showMessage(err.message || "تعذر حفظ التعديل", "error", editFormMessage);
        }
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

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && editLexiconModal && !editLexiconModal.classList.contains("hidden")) {
            closeEditLexiconModal();
        }
    });

    loadLexicon();
});