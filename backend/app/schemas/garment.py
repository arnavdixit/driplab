"""
Pydantic schemas for garment resources.
"""
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ALLOWED_STATUSES = {"pending", "processing", "ready", "failed"}


class GarmentPredictionResponse(BaseModel):
    """Metadata returned for a garment prediction (when available)."""

    category: str
    subcategory: Optional[str] = None
    attributes: dict[str, Any] | None = None
    bbox_x: Optional[int] = None
    bbox_y: Optional[int] = None
    bbox_width: Optional[int] = None
    bbox_height: Optional[int] = None
    detection_confidence: Optional[float] = None
    category_confidence: Optional[float] = None
    embedding_id: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class GarmentResponse(BaseModel):
    """Response model for garments (single item)."""

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
    prediction: Optional[GarmentPredictionResponse] = None

    model_config = ConfigDict(from_attributes=True)


class WardrobeListResponse(BaseModel):
    """Paginated wardrobe listing."""

    items: list[GarmentResponse]
    total: int
    limit: int
    offset: int


class WardrobeListFilters(BaseModel):
    """Query params for wardrobe listing."""

    status: Optional[str] = Field(default=None, description="Filter garments by status")
    category: Optional[str] = Field(default=None, description="Filter garments by predicted category")
    limit: int = Field(default=20, ge=1, le=100, description="Max items to return")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: Optional[str]) -> Optional[str]:
        if value and value not in ALLOWED_STATUSES:
            raise ValueError(f"status must be one of {sorted(ALLOWED_STATUSES)}")
        return value


class GarmentUpdateRequest(BaseModel):
    """Allowed updates for a garment."""

    custom_name: Optional[str] = Field(default=None, max_length=100)
    custom_notes: Optional[str] = None
    status: Optional[str] = None

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: Optional[str]) -> Optional[str]:
        if value and value not in ALLOWED_STATUSES:
            raise ValueError(f"status must be one of {sorted(ALLOWED_STATUSES)}")
        return value

    @model_validator(mode="after")
    def at_least_one_field(self) -> "GarmentUpdateRequest":
        if not any([self.custom_name is not None, self.custom_notes is not None, self.status is not None]):
            raise ValueError("At least one field must be provided")
        return self
