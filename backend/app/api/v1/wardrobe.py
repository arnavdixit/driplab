"""
Wardrobe-related API endpoints.
"""
import io
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session, joinedload

from backend.app.api.deps import get_current_user
from backend.app.core.config import settings
from backend.app.db.base import get_db
from backend.app.models.garment import Garment
from backend.app.models.garment_prediction import GarmentPrediction
from backend.app.models.user import User
from backend.app.schemas.garment import GarmentResponse, GarmentUpdateRequest, WardrobeListFilters, WardrobeListResponse
from backend.app.services import get_storage
from backend.app.services.storage import Storage

router = APIRouter(prefix="/wardrobe", tags=["wardrobe"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB


def _serialize_garment(garment: Garment, storage: Storage | None = None) -> GarmentResponse:
    """Shape a garment ORM object into the API response without triggering lazy loads."""
    prediction_obj = garment.__dict__.get("predictions")
    prediction = None
    if prediction_obj:
        prediction = {
            "category": prediction_obj.category,
            "subcategory": prediction_obj.subcategory,
            "attributes": prediction_obj.attributes,
            "bbox_x": prediction_obj.bbox_x,
            "bbox_y": prediction_obj.bbox_y,
            "bbox_width": prediction_obj.bbox_width,
            "bbox_height": prediction_obj.bbox_height,
            "detection_confidence": prediction_obj.detection_confidence,
            "category_confidence": prediction_obj.category_confidence,
            "embedding_id": prediction_obj.embedding_id,
        }

    def to_url(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        if storage:
            return storage.get_url(path)
        return path

    return GarmentResponse(
        id=garment.id,
        status=garment.status,
        original_image_path=to_url(garment.original_image_path),
        processed_image_path=to_url(garment.processed_image_path),
        thumbnail_path=to_url(garment.thumbnail_path),
        error_message=garment.error_message,
        custom_name=garment.custom_name,
        custom_notes=garment.custom_notes,
        created_at=garment.created_at,
        updated_at=garment.updated_at,
        prediction=prediction,
    )


def _delete_path(storage: Storage, path: Optional[str]) -> None:
    """Best-effort deletion for stored files (supports absolute or storage-relative paths)."""
    if not path:
        return

    target = Path(path)
    if target.is_absolute():
        try:
            target.unlink()
        except FileNotFoundError:
            return
    else:
        try:
            storage.delete(path)
        except (FileNotFoundError, ValueError):
            return


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
    storage: Storage = Depends(get_storage),
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

    # Save original
    original_relative = storage.save(data, directory="uploads", filename=file.filename)

    # Generate thumbnail
    try:
        image = Image.open(io.BytesIO(data))
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image file."
        ) from exc

    thumb = image.copy()
    thumb.thumbnail((320, 320))
    thumb_buffer = io.BytesIO()
    save_format = image.format or "JPEG"
    thumb.save(thumb_buffer, format=save_format)
    thumbnail_relative = storage.save(
        thumb_buffer.getvalue(), directory="thumbnails", filename=file.filename
    )

    garment = Garment(
        user_id=current_user.id,
        original_image_path=str(original_relative),
        thumbnail_path=str(thumbnail_relative),
        status="pending",
    )
    db.add(garment)
    db.commit()
    db.refresh(garment)

    return _serialize_garment(garment, storage=storage)


@router.get(
    "",
    response_model=WardrobeListResponse,
    summary="List garments with pagination and filters",
)
def list_garments(
    filters: WardrobeListFilters = Depends(),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
) -> WardrobeListResponse:
    query = db.query(Garment).filter(Garment.user_id == current_user.id)

    if filters.status:
        query = query.filter(Garment.status == filters.status)

    joined_prediction = False
    if filters.category:
        if db.bind and db.bind.dialect.name == "sqlite":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Category filter not supported with the SQLite test database.",
            )
        query = query.join(GarmentPrediction).filter(GarmentPrediction.category == filters.category)
        joined_prediction = True

    total = query.count()

    if db.bind and db.bind.dialect.name != "sqlite":
        query = query.options(joinedload(Garment.predictions))
    elif joined_prediction:
        query = query.options(joinedload(Garment.predictions))

    garments = (
        query.order_by(Garment.created_at.desc())
        .offset(filters.offset)
        .limit(filters.limit)
        .all()
    )

    items = [_serialize_garment(g, storage=storage) for g in garments]
    return WardrobeListResponse(items=items, total=total, limit=filters.limit, offset=filters.offset)


@router.get(
    "/{garment_id}",
    response_model=GarmentResponse,
    summary="Get a garment with prediction details",
)
def get_garment(
    garment_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
) -> GarmentResponse:
    query = db.query(Garment).filter(Garment.id == garment_id, Garment.user_id == current_user.id)
    if db.bind and db.bind.dialect.name != "sqlite":
        query = query.options(joinedload(Garment.predictions))

    garment = query.first()
    if not garment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Garment not found")

    return _serialize_garment(garment, storage=storage)


@router.patch(
    "/{garment_id}",
    response_model=GarmentResponse,
    summary="Update a garment's metadata",
)
def update_garment(
    garment_id: uuid.UUID,
    payload: GarmentUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
) -> GarmentResponse:
    garment = (
        db.query(Garment)
        .filter(Garment.id == garment_id, Garment.user_id == current_user.id)
        .first()
    )
    if not garment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Garment not found")

    if payload.custom_name is not None:
        garment.custom_name = payload.custom_name
    if payload.custom_notes is not None:
        garment.custom_notes = payload.custom_notes
    if payload.status is not None:
        garment.status = payload.status

    db.add(garment)
    db.commit()
    db.refresh(garment)

    return _serialize_garment(garment, storage=storage)


@router.delete(
    "/{garment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a garment and its files",
)
def delete_garment(
    garment_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
) -> None:
    garment = (
        db.query(Garment)
        .filter(Garment.id == garment_id, Garment.user_id == current_user.id)
        .first()
    )
    if not garment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Garment not found")

    original_path = garment.original_image_path
    processed_path = garment.processed_image_path
    thumbnail_path = garment.thumbnail_path

    db.delete(garment)
    db.commit()

    _delete_path(storage, original_path)
    _delete_path(storage, processed_path)
    _delete_path(storage, thumbnail_path)
