from __future__ import annotations

import json
import secrets
from urllib.parse import urlencode

import requests
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from core.config import get_settings


router = APIRouter(prefix="/youtube-auth", tags=["YouTube OAuth"])
templates = Jinja2Templates(directory="templates")

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
YOUTUBE_SCOPES = (
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
)


def _redirect_uri(request: Request) -> str:
    settings = get_settings()
    return settings.youtube_redirect_uri or str(request.url_for("youtube_auth_callback"))


def _render(request: Request, **context: object) -> HTMLResponse:
    response = templates.TemplateResponse(
        name="youtube_auth.html",
        request=request,
        context=context,
    )
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["X-Robots-Tag"] = "noindex, nofollow"
    return response


@router.get("", response_class=HTMLResponse)
async def youtube_auth_page(request: Request):
    settings = get_settings()
    configured = bool(settings.google_client_id and settings.google_client_secret)
    return _render(
        request,
        configured=configured,
        redirect_uri=_redirect_uri(request),
        scopes=YOUTUBE_SCOPES,
    )


@router.get("/start")
async def youtube_auth_start(request: Request):
    settings = get_settings()
    if not settings.google_client_id or not settings.google_client_secret:
        return _render(
            request,
            configured=False,
            redirect_uri=_redirect_uri(request),
            scopes=YOUTUBE_SCOPES,
            error="أضف GOOGLE_CLIENT_ID وGOOGLE_CLIENT_SECRET أولًا ثم أعد تشغيل التطبيق.",
        )

    state = secrets.token_urlsafe(32)
    request.session["youtube_oauth_state"] = state
    query = urlencode(
        {
            "client_id": settings.google_client_id,
            "redirect_uri": _redirect_uri(request),
            "response_type": "code",
            "scope": " ".join(YOUTUBE_SCOPES),
            "access_type": "offline",
            "include_granted_scopes": "true",
            "prompt": "consent",
            "state": state,
        }
    )
    return RedirectResponse(f"{GOOGLE_AUTH_URL}?{query}", status_code=302)


@router.get("/callback", name="youtube_auth_callback", response_class=HTMLResponse)
async def youtube_auth_callback(request: Request, code: str | None = None, state: str | None = None, error: str | None = None):
    settings = get_settings()
    expected_state = request.session.pop("youtube_oauth_state", None)

    if error:
        return _render(request, configured=True, redirect_uri=_redirect_uri(request), scopes=YOUTUBE_SCOPES, error=f"رفض Google الطلب: {error}")
    if not code or not state or not expected_state or not secrets.compare_digest(state, expected_state):
        return _render(request, configured=True, redirect_uri=_redirect_uri(request), scopes=YOUTUBE_SCOPES, error="فشل التحقق من جلسة OAuth. ابدأ التفويض من جديد.")
    if not settings.google_client_id or not settings.google_client_secret:
        return _render(request, configured=False, redirect_uri=_redirect_uri(request), scopes=YOUTUBE_SCOPES, error="إعدادات Google غير مكتملة.")

    try:
        token_response = requests.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": _redirect_uri(request),
                "grant_type": "authorization_code",
            },
            timeout=20,
        )
        token_response.raise_for_status()
        token_data = token_response.json()
    except (requests.RequestException, ValueError) as exc:
        detail = "تعذر استبدال رمز التفويض بتوكن. تأكد من Client ID وClient Secret وRedirect URI."
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            try:
                google_error = exc.response.json().get("error_description")
                if google_error:
                    detail = f"Google: {google_error}"
            except ValueError:
                pass
        return _render(request, configured=True, redirect_uri=_redirect_uri(request), scopes=YOUTUBE_SCOPES, error=detail)

    return _render(
        request,
        configured=True,
        redirect_uri=_redirect_uri(request),
        scopes=YOUTUBE_SCOPES,
        token_json=json.dumps(token_data, ensure_ascii=False, indent=2),
    )
