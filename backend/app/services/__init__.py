"""Service factories."""

from backend.app.services.storage import LocalStorage, Storage


def get_storage() -> Storage:
    """Provide a storage instance. Swappable for future backends."""
    return LocalStorage()
