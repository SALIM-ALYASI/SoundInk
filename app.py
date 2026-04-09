from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.v1.routes import router as api_router
from api.v1.preset_routes import router as preset_router
from api.v1.merge_routes import router as merge_router

app = FastAPI(title="APVA Next")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.include_router(api_router, prefix="/api/v1")
app.include_router(preset_router, prefix="/api/v1")
app.include_router(merge_router, prefix="/api/v1")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )