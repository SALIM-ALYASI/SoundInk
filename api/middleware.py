from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from core.logger import setup_logger

logger = setup_logger("API_Middleware")

def setup_error_handlers(app):
    """
    Overrides default FastAPI exception handlers to provide
    standardized JSON error responses.
    """
    
    @app.exception_handler(HTTPException)
    async def standard_http_exception_handler(request: Request, exc: HTTPException):
        logger.warning(f"HTTP Exception: {exc.detail} - Path: {request.url.path}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "code": exc.status_code,
                "message": exc.detail
            }
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled Server Error: {str(exc)} - Path: {request.url.path}")
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "code": 500,
                "message": "Internal Server Error. Please contact support."
            }
        )
