"""Interface for point cloud segmentation backbones."""

from typing import Protocol

import numpy as np


class PointSegmenter(Protocol):
    """Predict per-point class scores from point cloud features."""

    def predict(self, points: np.ndarray) -> np.ndarray:
        """Return per-point class scores or probabilities."""

