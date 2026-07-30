from __future__ import annotations

from fastapi import Header, HTTPException, status

from core.config import get_settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """
    حماية بسيطة لنقاط الوصول عبر مفتاح API ثابت.

    لو ما تم ضبط API_KEY بالبيئة، تبقى الحماية معطّلة (وضع التطوير المحلي).
    عند النشر على سيرفر عام، لازم تضبط API_KEY، وتُرسل قيمته من العميل
    برأس الطلب: X-API-Key.

    ملاحظة: هذا رادع أساسي فقط ضد الاستخدام العشوائي غير المصرّح، وليس بديلاً
    عن وضع الخدمة خلف reverse proxy بحماية إضافية (Basic Auth / VPN / IP
    allowlist) عند النشر الفعلي على الإنترنت.
    """
    settings = get_settings()

    if not settings.api_key:
        return

    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
