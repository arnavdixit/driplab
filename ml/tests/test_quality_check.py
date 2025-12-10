from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ml.ingestion.quality_check import ImageQualityChecker, QualityThresholds


def _save_image(path: Path, array: np.ndarray) -> Path:
    Image.fromarray(array.astype(np.uint8)).save(path, format="PNG")
    return path


def test_rejects_low_resolution(tmp_path: Path) -> None:
    small = np.full((100, 100, 3), 200, dtype=np.uint8)
    img_path = _save_image(tmp_path / "small.png", small)

    checker = ImageQualityChecker()
    result = checker.run_all(str(img_path))

    assert result["passed"] is False
    assert any("too small" in err.lower() for err in result["errors"])


def test_blur_warning(tmp_path: Path) -> None:
    flat = np.full((256, 256, 3), 128, dtype=np.uint8)
    img_path = _save_image(tmp_path / "flat.png", flat)

    checker = ImageQualityChecker()
    result = checker.run_all(str(img_path))

    assert any("blurry" in warn.lower() for warn in result["warnings"])


def test_exposure_warning(tmp_path: Path) -> None:
    dark = np.zeros((256, 256, 3), dtype=np.uint8)
    img_path = _save_image(tmp_path / "dark.png", dark)

    checker = ImageQualityChecker()
    result = checker.run_all(str(img_path))

    assert any("too_dark" in warn for warn in result["warnings"])


def test_clothing_confidence_flag(tmp_path: Path) -> None:
    plain = np.full((256, 256, 3), 240, dtype=np.uint8)
    img_path = _save_image(tmp_path / "plain.png", plain)

    checker = ImageQualityChecker()
    result = checker.run_all(str(img_path))

    assert any("clothing" in warn.lower() for warn in result["warnings"])


def test_passes_good_image(tmp_path: Path) -> None:
    patterned = np.zeros((256, 256, 3), dtype=np.uint8)
    patterned[32:224, 32:224] = (180, 50, 120)
    # Add grid lines to ensure edges/texture
    for i in range(32, 224, 24):
        patterned[i : i + 4, 32:224] = 255
        patterned[32:224, i : i + 4] = 60

    img_path = _save_image(tmp_path / "pattern.png", patterned)

    checker = ImageQualityChecker(QualityThresholds(clothing_threshold=0.5))
    result = checker.run_all(str(img_path))

    assert result["passed"] is True
    assert result["errors"] == []
    assert all("blurry" not in warn.lower() for warn in result["warnings"])
