"""
Garment model for uploaded clothing items.
"""
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Column, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.app.db.base import Base

if TYPE_CHECKING:
    from backend.app.models.user import User  # noqa: F401
    from backend.app.models.garment_prediction import GarmentPrediction  # noqa: F401
    from backend.app.models.garment_label import GarmentLabel  # noqa: F401


class Garment(Base):
    """
    Garment model representing an uploaded clothing item.
    
    Status values:
    - pending: Uploaded but not yet processed
    - processing: Currently being processed by ML pipeline
    - ready: Processing complete, ready for use
    - failed: Processing failed
    """

    __tablename__ = "garments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Image paths
    original_image_path = Column(String(500), nullable=False)
    processed_image_path = Column(String(500), nullable=True)
    thumbnail_path = Column(String(500), nullable=True)

    # Processing status
    status = Column(String(20), default="pending", nullable=False)
    # Values: pending, processing, ready, failed
    error_message = Column(Text, nullable=True)

    # User overrides (optional manual naming)
    custom_name = Column(String(100), nullable=True)
    custom_notes = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="garments")
    predictions = relationship(
        "GarmentPrediction",
        back_populates="garment",
        uselist=False,
        cascade="all, delete-orphan",
    )
    labels = relationship(
        "GarmentLabel", back_populates="garment", uselist=False, cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("idx_garments_user", "user_id"),
        Index("idx_garments_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<Garment(id={self.id}, user_id={self.user_id}, status={self.status})>"
