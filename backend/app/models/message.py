"""
Message model for chat messages.
"""
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Column, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.app.db.base import Base

if TYPE_CHECKING:
    from backend.app.models.conversation import Conversation  # noqa: F401


class Message(Base):
    """
    Message model for chat messages in conversations.
    
    Role values:
    - user: Message from the user
    - assistant: Message from the assistant/bot
    """

    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )

    role = Column(String(10), nullable=False)
    # Values: user, assistant
    content = Column(Text, nullable=False)

    # For assistant messages: which outfits were recommended
    recommended_outfit_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=True)

    # Constraints extracted from this message
    extracted_constraints = Column(JSONB, nullable=True)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

    # Indexes
    __table_args__ = (
        Index("idx_messages_conversation", "conversation_id"),
        Index("idx_messages_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Message(id={self.id}, conversation_id={self.conversation_id}, role={self.role})>"
