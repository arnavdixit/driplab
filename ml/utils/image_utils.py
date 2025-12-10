from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

PathLike = Union[str, Path]
ImageInput = Union[PathLike, bytes, bytearray]
ColorTuple = Tuple[int, int, int]


class ImageLoadError(ValueError):
    """Raised when an image cannot be loaded or decoded."""


def load_image(source: ImageInput, mode: str = "RGB") -> Image.Image:
    """
    Load an image from a path or bytes and convert to the requested color mode.

    Supports JPEG/PNG/WebP. Raises ImageLoadError for corrupted/unsupported inputs.
    """
    try:
        if isinstance(source, (bytes, bytearray)):
            with Image.open(io.BytesIO(source)) as img:
                return img.convert(mode)

        path = Path(source)
        with Image.open(path) as img:
            return img.convert(mode)
    except (UnidentifiedImageError, OSError) as exc:  # Pillow uses OSError for some decode errors
        raise ImageLoadError(f"Failed to load image: {exc}") from exc


def to_numpy(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image to an RGB numpy array (H, W, 3) with dtype uint8."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image, dtype=np.uint8)


def ensure_rgb(image_np: np.ndarray) -> np.ndarray:
    """
    Ensure an array is in RGB order. If assumed BGR, it will be flipped.

    This function returns a copy to avoid mutating the input.
    """
    if image_np.shape[-1] != 3:
        raise ValueError("Expected image with 3 channels")
    # Heuristic: assume input might be BGR; caller should know the source.
    return image_np[..., ::-1].copy()


def ensure_bgr(image_np: np.ndarray) -> np.ndarray:
    """
    Ensure an array is in BGR order by flipping RGB to BGR.

    This function returns a copy to avoid mutating the input.
    """
    if image_np.shape[-1] != 3:
        raise ValueError("Expected image with 3 channels")
    return image_np[..., ::-1].copy()


def resize_with_aspect(
    image_np: np.ndarray,
    target_size: int,
    pad_color: ColorTuple = (0, 0, 0),
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Resize while preserving aspect ratio and pad to a square canvas.

    Returns (resized_image, metadata) where metadata contains:
      - scale: scale factor applied relative to original size
      - pad: (top, bottom, left, right) padding in pixels
      - original_size: (height, width)
    """
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError("Expected image_np with shape (H, W, 3)")

    orig_h, orig_w = image_np.shape[:2]
    if orig_h == 0 or orig_w == 0:
        raise ValueError("Invalid image with zero dimension")

    scale = target_size / float(max(orig_h, orig_w))
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    # Resize using Pillow for consistency across formats
    resized_img = Image.fromarray(image_np.astype(np.uint8)).resize((new_w, new_h), Image.BILINEAR)
    resized_np = np.array(resized_img, dtype=np.uint8)

    canvas = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)

    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_right = target_size - new_w - pad_left

    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized_np

    meta = {
        "scale": scale,
        "pad": (pad_top, pad_bottom, pad_left, pad_right),
        "original_size": (orig_h, orig_w),
    }

    return canvas, meta


