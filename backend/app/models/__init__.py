"""
SQLAlchemy models for the fashion app.
"""
from backend.app.models.conversation import Conversation
from backend.app.models.garment import Garment
from backend.app.models.garment_label import GarmentLabel
from backend.app.models.garment_prediction import GarmentPrediction
from backend.app.models.message import Message
from backend.app.models.outfit import Outfit
from backend.app.models.outfit_feedback import OutfitFeedback
from backend.app.models.user import User
from backend.app.models.user_preferences import UserPreferences

__all__ = [
    "User",
    "Garment",
    "GarmentPrediction",
    "GarmentLabel",
    "Outfit",
    "OutfitFeedback",
    "UserPreferences",
    "Conversation",
    "Message",
]
