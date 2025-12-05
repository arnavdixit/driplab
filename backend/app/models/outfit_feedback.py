"""
OutfitFeedback model for user feedback on outfits.
"""
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Column, DateTime, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.app.db.base import Base

if TYPE_CHECKING:
    from backend.app.models.outfit import Outfit  # noqa: F401


class OutfitFeedback(Base):
    """
    OutfitFeedback model for user feedback on outfit recommendations.
    
    Action values:
    - like: User liked this outfit
    - dislike: User disliked this outfit
    - wear: User wore this outfit
    - save: User saved this outfit
    - skip: User skipped this outfit
    
    Reason values:
    - colors: Related to color choices
    - style: Related to style match
    - comfort: Related to comfort
    - occasion_match: Related to occasion appropriateness
    - other: Other reason
    """

    __tablename__ = "outfit_feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    outfit_id = Column(
        UUID(as_uuid=True), ForeignKey("outfits.id", ondelete="CASCADE"), nullable=False
    )
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    action = Column(String(20), nullable=False)
    # Values: like, dislike, wear, save, skip
    reason = Column(String(50), nullable=True)
    # Values: colors, style, comfort, occasion_match, other

    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    outfit = relationship("Outfit", back_populates="feedback")

    # Indexes
    __table_args__ = (
        Index("idx_feedback_outfit", "outfit_id"),
        Index("idx_feedback_user", "user_id"),
        Index("idx_feedback_action", "action"),
    )

    def __repr__(self) -> str:
        return f"<OutfitFeedback(id={self.id}, outfit_id={self.outfit_id}, action={self.action})>"
