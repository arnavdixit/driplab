"""
UserPreferences model for user style preferences.
"""
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Column, DateTime, Float, ForeignKey, String
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.app.db.base import Base

if TYPE_CHECKING:
    from backend.app.models.user import User  # noqa: F401


class UserPreferences(Base):
    """
    UserPreferences model for storing user style preferences.
    
    Used for personalization of recommendations.
    """

    __tablename__ = "user_preferences"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    # Style preferences
    preferred_styles = Column(ARRAY(String(50)), default=[], nullable=False)
    # Values: casual, preppy, minimalist, streetwear, classic, bohemian, etc.
    avoid_styles = Column(ARRAY(String(50)), default=[], nullable=False)

    # Colors
    favorite_colors = Column(ARRAY(String(30)), default=[], nullable=False)
    avoid_colors = Column(ARRAY(String(30)), default=[], nullable=False)

    # Fit
    preferred_fit = Column(String(20), default="regular", nullable=False)
    # Values: slim, regular, relaxed

    # Formality range (0-1)
    formality_min = Column(Float, default=0.0, nullable=False)
    formality_max = Column(Float, default=1.0, nullable=False)

    # Comfort vs style balance (0=comfort, 1=style)
    comfort_style_balance = Column(Float, default=0.5, nullable=False)

    # Learned weights (updated by ML)
    learned_weights = Column(JSONB, default={}, nullable=False)

    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="preferences")

    def __repr__(self) -> str:
        return f"<UserPreferences(id={self.id}, user_id={self.user_id})>"
