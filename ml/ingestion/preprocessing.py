from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from ml.utils.image_utils import (
    ImageInput,
    ImageLoadError,
    load_image,
    resize_with_aspect,
    to_numpy,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing targets."""

    yolo_size: int = 640
    classifier_size: int = 224
    pad_color: tuple[int, int, int] = (0, 0, 0)
    normalization: str = "unit"  # "unit" (0-1) or "imagenet"


def normalize(image_np: np.ndarray, mode: str = "unit") -> np.ndarray:
    """
    Normalize an RGB image.

    - unit: scale to [0, 1]
    - imagenet: scale then standardize with ImageNet stats
    """
    image = image_np.astype(np.float32)
    if mode == "unit":
        return image / 255.0
    if mode == "imagenet":
        scaled = image / 255.0
        return (scaled - IMAGENET_MEAN) / IMAGENET_STD
    raise ValueError(f"Unsupported normalization mode: {mode}")


def preprocess_image(
    source: ImageInput,
    *,
    config: PreprocessingConfig = PreprocessingConfig(),
    targets: Sequence[str] = ("yolo", "classifier"),
) -> Mapping[str, object]:
    """
    Load once and produce resized + normalized variants.

    Returns a dict with:
      - original: original RGB np.ndarray
      - yolo/classifier: {image, normalized, meta}
    """
    try:
        pil_image = load_image(source, mode="RGB")
    except ImageLoadError as exc:
        raise ValueError(str(exc)) from exc

    original_np = to_numpy(pil_image)
    results: dict[str, object] = {"original": original_np}

    target_set = set(targets)

    if "yolo" in target_set:
        resized, meta = resize_with_aspect(original_np, config.yolo_size, pad_color=config.pad_color)
        results["yolo"] = {
            "image": resized,
            "normalized": normalize(resized, mode=config.normalization),
            "meta": meta,
        }

    if "classifier" in target_set:
        resized, meta = resize_with_aspect(original_np, config.classifier_size, pad_color=config.pad_color)
        results["classifier"] = {
            "image": resized,
            "normalized": normalize(resized, mode=config.normalization),
            "meta": meta,
        }

    return results


