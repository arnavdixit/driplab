"""
Shared API dependencies.
"""
from uuid import UUID

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.base import get_db
from backend.app.models.user import User


def get_current_user(
    db: Session = Depends(get_db), authorization: str | None = Header(default=None)
) -> User:
    """
    Token-oriented user retrieval.

    Expects Authorization: Bearer <user_uuid>. Raises 401 if missing/invalid, 404 if user not found.
    """
    if not authorization:
        if settings.ALLOW_DEV_SINGLE_USER:
            user = db.query(User).filter(User.email == settings.DEV_SINGLE_USER_EMAIL).first()
            if user:
                return user
            dev_user = User(email=settings.DEV_SINGLE_USER_EMAIL)
            db.add(dev_user)
            db.commit()
            db.refresh(dev_user)
            return dev_user
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization token required",
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header",
        )

    token = authorization.split(" ", 1)[1].strip()
    try:
        user_id = UUID(token)
    except ValueError as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization token",
        ) from exc

    user = db.get(User, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return user
