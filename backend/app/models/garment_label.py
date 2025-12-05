"""
GarmentLabel model for user corrections.
"""
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Column, DateTime, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.app.db.base import Base

if TYPE_CHECKING:
    from backend.app.models.garment import Garment  # noqa: F401


class GarmentLabel(Base):
    """
    GarmentLabel model for user corrections/overrides of ML predictions.
    
    Source values:
    - user_created: User manually created this label
    - user_corrected: User corrected an ML prediction
    """

    __tablename__ = "garment_labels"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    garment_id = Column(
        UUID(as_uuid=True), ForeignKey("garments.id", ondelete="CASCADE"), nullable=False
    )

    category = Column(String(50), nullable=True)
    subcategory = Column(String(50), nullable=True)
    attributes = Column(JSONB, nullable=True)

    source = Column(String(20), nullable=False)
    # Values: user_created, user_corrected

    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    garment = relationship("Garment", back_populates="labels")

    # Indexes
    __table_args__ = (Index("idx_labels_garment", "garment_id"),)

    def __repr__(self) -> str:
        return f"<GarmentLabel(id={self.id}, garment_id={self.garment_id}, source={self.source})>"
