from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from ml.utils.image_utils import ImageLoadError, load_image, to_numpy


@dataclass
class QualityThresholds:
    """Thresholds controlling quality checks."""

    min_size: int = 224
    blur_variance: float = 100.0
    exposure_bounds: Tuple[int, int] = (40, 220)
    clothing_threshold: float = 0.7


class ImageQualityChecker:
    """
    Run heuristic quality checks before ingestion.

    Output schema:
        {
            "passed": bool,
            "warnings": list[str],
            "errors": list[str],
            "scores": {
                "resolution": int,   # shortest side in pixels
                "blur": float,       # Laplacian variance
                "exposure": float,   # mean brightness (0-255)
                "clothing": float,   # 0-1 heuristic confidence
            },
        }
    """

    def __init__(self, thresholds: QualityThresholds | None = None) -> None:
        self.thresholds = thresholds or QualityThresholds()

    def check_resolution(self, image: np.ndarray, min_size: int | None = None) -> tuple[bool, int]:
        """Check that shortest side meets the minimum size."""
        height, width = image.shape[:2]
        min_side = min(height, width)
        required = min_size if min_size is not None else self.thresholds.min_size
        return min_side >= required, min_side

    def check_blur(self, image: np.ndarray, threshold: float | None = None) -> tuple[bool, float]:
        """Compute Laplacian variance; low variance implies blur."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        limit = threshold if threshold is not None else self.thresholds.blur_variance
        return variance >= limit, variance

    def check_exposure(self, image: np.ndarray, bounds: Tuple[int, int] | None = None) -> tuple[bool, str, float]:
        """Assess exposure by mean brightness against configured bounds."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mean_brightness = float(gray.mean())
        low, high = bounds if bounds is not None else self.thresholds.exposure_bounds

        if mean_brightness < low:
            return False, "too_dark", mean_brightness
        if mean_brightness > high:
            return False, "too_bright", mean_brightness
        return True, "ok", mean_brightness

    def clothing_confidence(self, image: np.ndarray) -> float:
        """
        Estimate clothing presence via texture and color heuristics.

        Features:
        - Edge density from Sobel gradients
        - Saturation coverage in HSV space
        - Foreground occupancy from luminance deviation
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        sat = hsv[..., 1]
        saturation_coverage = float((sat > 35).mean())

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        edge_density = float((grad_mag > 40).mean())

        deviation_mask = np.abs(gray - gray.mean()) > 10
        foreground_ratio = float(deviation_mask.mean())

        score = 0.5 * edge_density + 0.3 * saturation_coverage + 0.2 * foreground_ratio
        return float(max(0.0, min(1.0, score)))

    def run_all(
        self,
        image_path: str,
        *,
        min_size: int | None = None,
        blur_thresh: float | None = None,
        exposure_bounds: Tuple[int, int] | None = None,
        clothing_thresh: float | None = None,
    ) -> dict:
        """
        Execute all quality checks on an image path.

        Returns a structured dict with pass/fail, warnings/errors, and scores.
        """
        thresholds = QualityThresholds(
            min_size=min_size or self.thresholds.min_size,
            blur_variance=blur_thresh or self.thresholds.blur_variance,
            exposure_bounds=exposure_bounds or self.thresholds.exposure_bounds,
            clothing_threshold=clothing_thresh or self.thresholds.clothing_threshold,
        )

        try:
            pil_image = load_image(image_path, mode="RGB")
        except ImageLoadError as exc:
            raise ValueError(str(exc)) from exc

        image_np = to_numpy(pil_image)

        result: dict[str, object] = {
            "passed": True,
            "warnings": [],
            "errors": [],
            "scores": {
                "resolution": 0,
                "blur": 0.0,
                "exposure": 0.0,
                "clothing": 0.0,
            },
        }

        is_big_enough, min_side = self.check_resolution(image_np, thresholds.min_size)
        result["scores"]["resolution"] = min_side
        if not is_big_enough:
            result["passed"] = False
            result["errors"].append(f"Image too small (min {thresholds.min_size}px on shortest side)")
            return result

        is_sharp, blur_score = self.check_blur(image_np, thresholds.blur_variance)
        result["scores"]["blur"] = blur_score
        if not is_sharp:
            result["warnings"].append(f"Image may be blurry (variance={blur_score:.1f})")

        is_exposed, exposure_status, mean_brightness = self.check_exposure(
            image_np, thresholds.exposure_bounds
        )
        result["scores"]["exposure"] = mean_brightness
        if not is_exposed:
            result["warnings"].append(f"Image is {exposure_status}")

        clothing_score = self.clothing_confidence(image_np)
        result["scores"]["clothing"] = clothing_score
        if clothing_score < thresholds.clothing_threshold:
            result["warnings"].append(
                f"Image may not contain clothing (confidence={clothing_score:.2f})"
            )

        return result
