"""
Pydantic schemas for garment resources.
"""
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class GarmentResponse(BaseModel):
    """Response model for garments."""

    id: UUID
    status: str
    original_image_path: str
    processed_image_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    error_message: Optional[str] = None
    custom_name: Optional[str] = None
    custom_notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
