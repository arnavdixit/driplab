"""
User model.
"""
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Column, DateTime, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.app.db.base import Base

if TYPE_CHECKING:
    from backend.app.models.garment import Garment  # noqa: F401
    from backend.app.models.user_preferences import UserPreferences  # noqa: F401
    from backend.app.models.outfit import Outfit  # noqa: F401
    from backend.app.models.conversation import Conversation  # noqa: F401


class User(Base):
    """
    User model for authentication and user data.
    
    For MVP, we'll use a single-user approach (no authentication).
    Later, we can add password_hash and proper auth.
    """

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=True)  # NULL for single-user MVP
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    garments = relationship("Garment", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship(
        "UserPreferences", back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    outfits = relationship("Outfit", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email})>"
