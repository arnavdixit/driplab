"""
Conversation model for chat sessions.
"""
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Column, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.app.db.base import Base

if TYPE_CHECKING:
    from backend.app.models.user import User  # noqa: F401
    from backend.app.models.message import Message  # noqa: F401


class Conversation(Base):
    """
    Conversation model for chat sessions with the outfit recommendation bot.
    
    Stores active constraints and excluded items for the current session.
    """

    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Current session constraints
    active_constraints = Column(JSONB, default={}, nullable=False)

    # Excluded items for this session
    excluded_item_ids = Column(ARRAY(UUID(as_uuid=True)), default=[], nullable=False)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (Index("idx_conversations_user", "user_id"),)

    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, user_id={self.user_id})>"
