from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ml.ingestion.preprocessing import (
    PreprocessingConfig,
    normalize,
    preprocess_image,
)
from ml.utils.image_utils import ImageLoadError, load_image, resize_with_aspect


def _make_image(path: Path, size: tuple[int, int], color: tuple[int, int, int], fmt: str) -> Path:
    image = Image.new("RGB", size, color=color)
    image.save(path, format=fmt)
    return path


def test_load_image_from_path_and_bytes(tmp_path: Path) -> None:
    png_path = _make_image(tmp_path / "sample.png", (64, 32), (10, 20, 30), fmt="PNG")
    webp_path = _make_image(tmp_path / "sample.webp", (40, 80), (15, 25, 35), fmt="WEBP")

    # Path load
    img_png = load_image(png_path)
    assert img_png.size == (64, 32)

    # Bytes load
    with webp_path.open("rb") as fh:
        img_webp = load_image(fh.read())
    assert img_webp.size == (40, 80)


def test_load_image_rejects_corrupted_bytes() -> None:
    with pytest.raises(ImageLoadError):
        load_image(b"not-an-image")


def test_resize_with_aspect_padding() -> None:
    # Wide image -> expect vertical padding
    image_np = np.full((100, 200, 3), 255, dtype=np.uint8)
    resized, meta = resize_with_aspect(image_np, target_size=224, pad_color=(0, 0, 0))

    assert resized.shape == (224, 224, 3)
    assert meta["original_size"] == (100, 200)
    assert meta["pad"] == (56, 56, 0, 0)  # top, bottom, left, right

    # Check that padding is black and center region remains white
    assert resized[:10].max() == 0  # top padding
    assert resized[80:140, 10:214].min() == 255  # central area stays original


def test_normalize_unit_and_imagenet() -> None:
    image_np = np.full((2, 2, 3), 255, dtype=np.uint8)

    unit = normalize(image_np, mode="unit")
    assert unit.dtype == np.float32
    assert unit.max() == pytest.approx(1.0)

    imagenet = normalize(image_np, mode="imagenet")
    # For white image, (1 - mean) / std
    expected = (1.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    assert np.allclose(imagenet[0, 0], expected, atol=1e-4)


def test_preprocess_image_returns_targets(tmp_path: Path) -> None:
    jpeg_path = _make_image(tmp_path / "sample.jpg", (120, 60), (5, 15, 25), fmt="JPEG")
    config = PreprocessingConfig(yolo_size=128, classifier_size=64, normalization="unit")

    result = preprocess_image(jpeg_path, config=config, targets=("yolo", "classifier"))

    assert "original" in result
    assert result["yolo"]["image"].shape == (128, 128, 3)
    assert result["classifier"]["image"].shape == (64, 64, 3)
    assert result["yolo"]["normalized"].max() <= 1.0
    assert result["classifier"]["normalized"].max() <= 1.0


