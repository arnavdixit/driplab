"""
Storage abstraction with a local filesystem implementation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from backend.app.core.config import settings


class Storage(ABC):
    """Abstract storage interface."""

    @abstractmethod
    def save(self, data: BinaryIO | bytes, directory: str = "uploads", filename: str | None = None) -> str:
        """Persist data under the given directory and return the relative path."""

    @abstractmethod
    def get_url(self, path: str) -> str:
        """Return a retrievable URL for a stored path."""

    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete a stored file if it exists."""


class LocalStorage(Storage):
    """
    Local filesystem storage under a runtime root.

    Directories:
    - uploads (default)
    - processed
    - thumbnails
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else Path(settings.UPLOAD_DIR).parent
        self.directories = {
            "uploads": self.root / Path(settings.UPLOAD_DIR).name,
            "processed": self.root / Path(settings.PROCESSED_DIR).name,
            "thumbnails": self.root / "thumbnails",
        }
        for directory in self.directories.values():
            directory.mkdir(parents=True, exist_ok=True)

    def save(self, data: BinaryIO | bytes, directory: str = "uploads", filename: str | None = None) -> str:
        target_dir = self.directories.get(directory, self.root / directory)
        target_dir.mkdir(parents=True, exist_ok=True)

        ext = Path(filename).suffix if filename else ""
        unique_name = f"{uuid4().hex}{ext}"
        relative_path = Path(directory) / unique_name
        absolute_path = self._to_absolute(relative_path)

        payload = data if isinstance(data, bytes) else data.read()
        with absolute_path.open("wb") as file:
            file.write(payload)

        return str(relative_path)

    def get_url(self, path: str) -> str:
        """
        Return a web-accessible URL for a stored path.

        For local storage we serve files via FastAPI StaticFiles at /media,
        so we convert the stored relative path (e.g., uploads/abc.jpg) into
        /media/uploads/abc.jpg.
        """
        relative = Path(path)
        return f"/media/{relative.as_posix()}"

    def delete(self, path: str) -> None:
        absolute_path = self._to_absolute(Path(path))
        try:
            absolute_path.unlink()
        except FileNotFoundError:
            return

    def _to_absolute(self, relative_path: Path) -> Path:
        """Resolve a relative path safely within the root."""
        absolute = (self.root / relative_path).resolve()
        if self.root.resolve() not in absolute.parents and self.root.resolve() != absolute:
            raise ValueError("Path traversal detected")
        return absolute
