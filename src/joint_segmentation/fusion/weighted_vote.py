"""Simple weighted prediction fusion placeholder."""

import numpy as np


def fuse_scores(
    point_scores: np.ndarray,
    projected_scores: np.ndarray,
    point_weight: float = 0.5,
    image_weight: float = 0.5,
) -> np.ndarray:
    """Combine aligned point and projected image scores."""
    return (point_weight * point_scores) + (image_weight * projected_scores)

