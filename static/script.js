document.addEventListener("DOMContentLoaded", () => {
    // Generate Elements
    const generateBtn = document.getElementById("generate-btn");
    const textInput = document.getElementById("text-input");
    const spinner = document.getElementById("loading-spinner");
    const btnText = document.querySelector(".btn-text");

    // Result Elements
    const resultSection = document.getElementById("result-section");
    const normalPlayer = document.getElementById("normal-player");
    const cinematicPlayer = document.getElementById("cinematic-player");
    const normalDownload = document.getElementById("normal-download");
    const cinematicDownload = document.getElementById("cinematic-download");
    const sessionIdInput = document.getElementById("current-session-id");
    const segmentsList = document.getElementById("segments-list");

    // Lexicon Elements
    const lexOrig = document.getElementById("lex-orig");
    const lexCorr = document.getElementById("lex-corr");
    const addLexiconBtn = document.getElementById("add-lexicon-btn");
    const lexiconMsg = document.getElementById("lexicon-msg");

    const errorMsg = document.getElementById("error-message");

    // --- 0. Authentication ---
    const authModal = document.getElementById("auth-modal");
    const showLoginBtn = document.getElementById("show-login-btn");
    const loginSubmit = document.getElementById("login-submit");
    const registerSubmit = document.getElementById("register-submit");
    const authEmail = document.getElementById("auth-email");
    const authPassword = document.getElementById("auth-password");
    const authError = document.getElementById("auth-error");
    const authSection = document.getElementById("auth-section");
    const userSection = document.getElementById("user-section");
    const navTokens = document.getElementById("nav-tokens");

    function getToken() { return localStorage.getItem("auth_token"); }
    function setToken(t) { localStorage.setItem("auth_token", t); }

    async function checkAuth() {
        const token = getToken();
        if (!token) return;
        try {
            const res = await fetch("/api/v1/auth/me", { headers: { "Authorization": `Bearer ${token}` } });
            if (res.ok) {
                const user = await res.json();
                authSection.classList.add("hidden");
                userSection.classList.remove("hidden");
                navTokens.textContent = user.token_balance.toLocaleString() + " حرف";
                authModal.classList.add("hidden");
            } else {
                localStorage.removeItem("auth_token");
            }
        } catch (e) { console.error(e); }
    }

    checkAuth(); // Run on load

    showLoginBtn.addEventListener("click", () => authModal.classList.toggle("hidden"));

    async function handleAuth(action) {
        const email = authEmail.value;
        const password = authPassword.value;
        if (!email || !password) return authError.textContent = "الرجاء إدخال الإيميل وكلمة المرور", authError.classList.remove("hidden");

        authError.classList.add("hidden");
        loginSubmit.disabled = true; registerSubmit.disabled = true;

        try {
            let res;
            if (action === "register") {
                res = await fetch("/api/v1/auth/register", {
                    method: "POST", headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ email, password })
                });
            } else {
                const formData = new URLSearchParams();
                formData.append('username', email);
                formData.append('password', password);
                res = await fetch("/api/v1/auth/login", {
                    method: "POST", headers: { "Content-Type": "application/x-www-form-urlencoded" },
                    body: formData
                });
            }

            const data = await res.json();
            if (res.ok) {
                if (action === "register") {
                    authError.textContent = "تم التسجيل بنجاح! رجاءً قم بتسجيل الدخول.";
                    authError.style.color = "#4ade80";
                    authError.classList.remove("hidden");
                } else {
                    setToken(data.access_token);
                    await checkAuth();
                }
            } else {
                authError.textContent = data.detail || "حدث خطأ";
                authError.style.color = "#f87171";
                authError.classList.remove("hidden");
            }
        } catch (e) {
            authError.textContent = "خطأ في الاتصال بالسيرفر";
            authError.classList.remove("hidden");
        }
        loginSubmit.disabled = false; registerSubmit.disabled = false;
    }

    loginSubmit.addEventListener("click", () => handleAuth("login"));
    registerSubmit.addEventListener("click", () => handleAuth("register"));

    // --- 1. Add Lexicon Entry ---
    addLexiconBtn.addEventListener("click", async () => {
        const orig = lexOrig.value.trim();
        const corr = lexCorr.value.trim();
        if (!orig || !corr) return;

        try {
            const res = await fetch("/api/v1/lexicon", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${getToken()}`
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

    // --- 2. Main Generation ---
    generateBtn.addEventListener("click", () => startProcess("/api/v1/generate", { text: textInput.value.trim() }));

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
            const headers = {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${getToken()}`
            };

            const response = await fetch(url, {
                method: "POST",
                headers: headers,
                body: JSON.stringify(bodyData)
            });

            const data = await response.json();

            if (response.status === 401 || response.status === 402) {
                throw new Error("يجب تسجيل الدخول / رصيدك لا يكفي.");
            }
            if (!response.ok) throw new Error(data.detail || "خطأ غير متوقع من الخادم");

            pollStatus(data.task_id);
            // Deduct locally for fast UI update
            if (data.tokens_remaining) {
                navTokens.textContent = data.tokens_remaining.toLocaleString() + " حرف";
            }
        } catch (err) {
            showError(err.message);
            setLoading(false);
        }
    }

    async function pollStatus(taskId) {
        try {
            const statusRes = await fetch(`/api/v1/status/${taskId}`);
            const statusData = await statusRes.json();

            if (statusRes.status !== 200) throw new Error("خطأ في التحقق من حالة الطلب");

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

        normalDownload.href = data.normal_url;
        cinematicDownload.href = data.cinematic_url;

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
