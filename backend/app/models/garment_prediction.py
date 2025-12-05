"""
GarmentPrediction model for ML-generated metadata.
"""
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Column, DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.app.db.base import Base

if TYPE_CHECKING:
    from backend.app.models.garment import Garment  # noqa: F401


class GarmentPrediction(Base):
    """
    GarmentPrediction model storing ML-generated metadata for garments.
    
    Attributes JSONB structure:
    {
        "color_primary": "navy",
        "color_secondary": "white",
        "pattern": "solid",  # solid, striped, plaid, floral, graphic
        "fit": "slim",  # slim, regular, relaxed, oversized
        "formality": 0.7,  # 0.0 (very casual) to 1.0 (very formal)
        "seasons": ["fall", "winter"],
        "style_tags": ["preppy", "classic"],
        "material": "cotton"
    }
    """

    __tablename__ = "garment_predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    garment_id = Column(
        UUID(as_uuid=True), ForeignKey("garments.id", ondelete="CASCADE"), nullable=False
    )

    # Detection bounding box
    bbox_x = Column(Integer, nullable=True)
    bbox_y = Column(Integer, nullable=True)
    bbox_width = Column(Integer, nullable=True)
    bbox_height = Column(Integer, nullable=True)
    detection_confidence = Column(Float, nullable=True)

    # Classification
    category = Column(String(50), nullable=False)
    # Values: t-shirt, shirt, blouse, sweater, hoodie, jacket, coat,
    #         blazer, jeans, pants, shorts, skirt, dress, sneakers,
    #         shoes, boots, sandals, belt, tie, hat, bag
    category_confidence = Column(Float, nullable=True)
    subcategory = Column(String(50), nullable=True)
    # Values: polo, henley, oxford, flannel, cardigan, bomber, etc.

    # Attributes (JSONB for flexibility)
    attributes = Column(JSONB, nullable=False, default={})

    # Reference to ChromaDB embedding
    embedding_id = Column(String(100), nullable=True)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    garment = relationship("Garment", back_populates="predictions")

    # Indexes
    __table_args__ = (
        Index("idx_predictions_garment", "garment_id"),
        Index("idx_predictions_category", "category"),
        Index("idx_predictions_attributes", "attributes", postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return f"<GarmentPrediction(id={self.id}, garment_id={self.garment_id}, category={self.category})>"
