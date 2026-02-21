document.addEventListener("DOMContentLoaded", () => {
    const generateBtn = document.getElementById("generate-btn");
    const textInput = document.getElementById("text-input");
    const spinner = document.getElementById("loading-spinner");
    const btnText = document.querySelector(".btn-text");
    const resultSection = document.getElementById("result-section");
    const audioPlayer = document.getElementById("audio-player");
    const downloadBtn = document.getElementById("download-btn");
    const errorMsg = document.getElementById("error-message");

    generateBtn.addEventListener("click", async () => {
        const text = textInput.value.trim();

        if (!text) {
            showError("الرجاء إدخال النص أولاً!");
            return;
        }

        // Set Loading State
        setLoading(true);
        hideError();
        resultSection.classList.add("hidden");

        try {
            const response = await fetch("/api/synthesize", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "خطأ غير متوقع من الخادم");
            }

            // Success State
            // adding timestamp to url to bust browser cache
            const audioUrl = `/api/audio?t=${new Date().getTime()}`;
            audioPlayer.src = audioUrl;
            downloadBtn.href = audioUrl;
            resultSection.classList.remove("hidden");

            // Auto-play attempt
            setTimeout(() => {
                audioPlayer.play().catch(e => console.log("User interaction needed to play audio"));
            }, 500);

        } catch (err) {
            showError(err.message);
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        if (isLoading) {
            generateBtn.disabled = true;
            textInput.disabled = true;
            spinner.classList.remove("hidden");
            btnText.textContent = "جاري التوليد...";
        } else {
            generateBtn.disabled = false;
            textInput.disabled = false;
            spinner.classList.add("hidden");
            btnText.textContent = "توليد الصوت";
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
