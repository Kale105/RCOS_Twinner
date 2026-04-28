"""Projection interfaces for mapping image labels into point cloud space."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProjectionResult:
    """Projected image labels aligned to point indices."""

    point_indices: np.ndarray
    class_scores: np.ndarray


class ImageToPointProjector:
    """Placeholder projector for calibrated image-to-point predictions."""

    def project(self, points: np.ndarray, image_scores: np.ndarray) -> ProjectionResult:
        raise NotImplementedError("Projection logic will be added after calibration format is chosen.")

