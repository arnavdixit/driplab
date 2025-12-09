"""
FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend.app.api.v1 import api_router
from backend.app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Wardrobe management and outfit recommendation API",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

# CORS middleware
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Fashion AI API",
        "version": settings.VERSION,
        "docs": "/docs",
        "api": settings.API_V1_STR,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get(f"{settings.API_V1_STR}/docs")
async def redirect_api_docs():
    """Redirect /api/v1/docs to /docs."""
    return RedirectResponse(url="/docs")
