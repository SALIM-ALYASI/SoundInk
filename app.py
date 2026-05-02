from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.v1.routes import router as api_router
from api.v1.preset_routes import router as preset_router
from api.v1.merge_routes import router as merge_router

app = FastAPI(
    title="APVA Next",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

app.include_router(api_router, prefix="/api/v1", tags=["Main API"])
app.include_router(preset_router, prefix="/api/v1", tags=["Presets"])
app.include_router(merge_router, prefix="/api/v1", tags=["Merge"])


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        name="index.html",
        request=request,
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "APVA Next"
    }