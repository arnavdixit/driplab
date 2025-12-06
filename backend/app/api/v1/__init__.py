"""
API v1 router.
"""
from fastapi import APIRouter

from backend.app.api.v1 import wardrobe

api_router = APIRouter()

api_router.include_router(wardrobe.router)


@api_router.get("/")
async def api_root():
    """API root endpoint."""
    return {"message": "Fashion AI API v1"}
