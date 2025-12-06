from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import Session, sessionmaker

from backend.app.api.deps import get_current_user
from backend.app.core.config import settings
from backend.app.db.base import Base, get_db
from backend.app.main import app
from backend.app.models import garment, user
from backend.app.models.garment import Garment
from backend.app.models.user import User


# Test database (in-memory SQLite)
engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    future=True,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(autouse=True)
def create_test_db() -> Generator[None, None, None]:
    # Limit table creation to models required for these tests to avoid unsupported
    # PostgreSQL-specific column types (e.g., JSONB) on SQLite.
    tables = [User.__table__, Garment.__table__]
    Base.metadata.create_all(bind=engine, tables=tables)
    yield
    Base.metadata.drop_all(bind=engine, tables=tables)


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def temp_upload_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    original_upload_dir = settings.UPLOAD_DIR
    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(upload_dir))
    yield upload_dir
    monkeypatch.setattr(settings, "UPLOAD_DIR", original_upload_dir)


@pytest.fixture
def client(db_session: Session, temp_upload_dir: Path) -> Generator[TestClient, None, None]:
    # Ensure models are imported so metadata is populated
    _ = garment, user

    def override_get_db() -> Generator[Session, None, None]:
        try:
            yield db_session
        finally:
            pass

    def override_get_current_user() -> User:
        existing = db_session.query(User).first()
        if existing:
            return existing
        new_user = User(email="test@example.com")
        db_session.add(new_user)
        db_session.commit()
        db_session.refresh(new_user)
        return new_user

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user

    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides = {}


def test_upload_success(client: TestClient, db_session: Session, temp_upload_dir: Path) -> None:
    files = {
        "file": ("test.png", b"\x89PNG\r\n\x1a\n", "image/png"),
    }

    response = client.post("/api/v1/wardrobe/upload", files=files)

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "pending"
    assert "id" in body
    assert db_session.query(Garment).count() == 1
    saved_files = list(temp_upload_dir.glob("*"))
    assert len(saved_files) == 1


def test_upload_rejects_invalid_type(client: TestClient, db_session: Session) -> None:
    files = {
        "file": ("note.txt", b"not an image", "text/plain"),
    }

    response = client.post("/api/v1/wardrobe/upload", files=files)

    assert response.status_code == 400
    assert db_session.query(Garment).count() == 0


def test_upload_rejects_too_large(client: TestClient, db_session: Session) -> None:
    oversized = bytes((10 * 1024 * 1024) + 1)
    files = {
        "file": ("big.png", oversized, "image/png"),
    }

    response = client.post("/api/v1/wardrobe/upload", files=files)

    assert response.status_code == 413
    assert db_session.query(Garment).count() == 0
