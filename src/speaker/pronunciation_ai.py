from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Optional


DEFAULT_MODEL = "llama3.2:3b"


@dataclass
class PronunciationAIResult:
    original_text: str
    improved_text: str
    model: str
    success: bool
    error: Optional[str] = None


def build_pronunciation_prompt(text: str) -> str:
    return f"""
أنت مساعد متخصص في تحسين النص العربي ليكون مناسبًا لتحويله إلى صوت واضح وطبيعي.

المطلوب:
1) لا تغيّر المعنى.
2) حسّن النص فقط من أجل النطق.
3) صحّح كتابة الكلمات الأجنبية لتُنطق بشكل أوضح بالعربية عند الحاجة.
4) أضف تشكيلًا خفيفًا فقط عند الضرورة لتحسين النطق.
5) قسّم الجمل الطويلة إذا كانت ستسبب صعوبة في النطق.
6) لا تضف شرحًا.
7) أعد النص النهائي فقط.

النص:
{text}
""".strip()


def ollama_is_available() -> bool:
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def improve_pronunciation_text(
    text: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 120,
) -> PronunciationAIResult:
    clean_text = (text or "").strip()
    if not clean_text:
        return PronunciationAIResult(
            original_text=text,
            improved_text=text,
            model=model,
            success=False,
            error="Empty text provided.",
        )

    if not ollama_is_available():
        return PronunciationAIResult(
            original_text=text,
            improved_text=text,
            model=model,
            success=False,
            error="Ollama is not available on this system.",
        )

    prompt = build_pronunciation_prompt(clean_text)

    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            error_msg = (result.stderr or result.stdout or "Unknown Ollama error").strip()
            return PronunciationAIResult(
                original_text=text,
                improved_text=text,
                model=model,
                success=False,
                error=error_msg,
            )

        improved = (result.stdout or "").strip()

        if not improved:
            return PronunciationAIResult(
                original_text=text,
                improved_text=text,
                model=model,
                success=False,
                error="Ollama returned empty output.",
            )

        return PronunciationAIResult(
            original_text=text,
            improved_text=improved,
            model=model,
            success=True,
            error=None,
        )

    except subprocess.TimeoutExpired:
        return PronunciationAIResult(
            original_text=text,
            improved_text=text,
            model=model,
            success=False,
            error=f"Ollama request timed out after {timeout} seconds.",
        )
    except Exception as exc:
        return PronunciationAIResult(
            original_text=text,
            improved_text=text,
            model=model,
            success=False,
            error=str(exc),
        )


def improve_pronunciation_or_fallback(
    text: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 120,
) -> str:
    result = improve_pronunciation_text(
        text=text,
        model=model,
        timeout=timeout,
    )

    if result.success and result.improved_text.strip():
        return result.improved_text.strip()

    return text