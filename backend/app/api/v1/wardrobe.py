"""
Wardrobe-related API endpoints.
"""
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from backend.app.api.deps import get_current_user
from backend.app.core.config import settings
from backend.app.db.base import get_db
from backend.app.models.garment import Garment
from backend.app.models.user import User
from backend.app.schemas.garment import GarmentResponse

router = APIRouter(prefix="/wardrobe", tags=["wardrobe"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB


@router.post(
    "/upload",
    response_model=GarmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a garment image",
)
async def upload_garment_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> GarmentResponse:
    """
    Upload a garment image, save it locally, and create a pending garment record.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Allowed: JPEG, PNG, WebP.",
        )

    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file extension. Allowed: .jpg, .jpeg, .png, .webp.",
        )

    data = await file.read()
    if len(data) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large. Max size is 10MB.",
        )

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4()}{extension}"
    file_path = upload_dir / filename
    file_path.write_bytes(data)

    garment = Garment(
        user_id=current_user.id,
        original_image_path=str(file_path),
        status="pending",
    )
    db.add(garment)
    db.commit()
    db.refresh(garment)

    return GarmentResponse.model_validate(garment)
