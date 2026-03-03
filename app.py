import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add core and src to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))
sys.path.append(str(BASE_DIR))

from core.config import settings
from api.middleware import setup_error_handlers
from api.v1.routes import router as api_v1_router
from core.database import Base, engine

# Create SQLite Database Tables on Startup
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Mount Static and Templates
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Setup global exception handlers for standardized JSON responses
setup_error_handlers(app)

# Include standard REST API versioned routes
app.include_router(api_v1_router, prefix="/api/v1", tags=["v1"])

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
