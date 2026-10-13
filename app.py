from __future__ import annotations

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.v1.routes import router as api_router
from api.v1.preset_routes import router as preset_router
from api.v1.merge_routes import router as merge_router
from api.v1.job_routes import router as job_router
from core.config import get_settings
from core.security import require_api_key

settings = get_settings()

app = FastAPI(
    title="APVA Next",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

_api_dependencies = [Depends(require_api_key)]

app.include_router(api_router, prefix="/api/v1", tags=["Main API"], dependencies=_api_dependencies)
app.include_router(preset_router, prefix="/api/v1", tags=["Presets"], dependencies=_api_dependencies)
app.include_router(merge_router, prefix="/api/v1", tags=["Merge"], dependencies=_api_dependencies)
app.include_router(job_router, prefix="/api/v1", tags=["Async TTS Jobs"], dependencies=_api_dependencies)


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )
