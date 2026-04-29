"""Joint fusion of camera projection and point-model segmentation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PredictionInputs:
    """Per-point predictions loaded from an `.npz` output file."""

    labels: np.ndarray
    scores: np.ndarray | None


@dataclass(frozen=True)
class JointSegmentationResult:
    """Fused segmentation output."""

    labels: np.ndarray
    scores: np.ndarray | None
    source: np.ndarray
    point_weight: float
    image_weight: float


@dataclass(frozen=True)
class JointSegmentationSummary:
    """Compact summary of a fused segmentation."""

    total_point_count: int
    fused_point_count: int
    point_model_count: int
    image_projection_count: int
    both_sources_count: int
    label_counts: dict[int, int]


def load_prediction_inputs(path: str | Path, point_count: int) -> PredictionInputs:
    """Load labels and optional scores from a projection/model prediction `.npz`."""
    prediction = np.load(path)
    labels = _load_labels(prediction, point_count)
    scores = _load_scores(prediction, point_count)
    return PredictionInputs(labels=labels, scores=scores)


def fuse_point_and_image_predictions(
    point_prediction: PredictionInputs,
    image_prediction: PredictionInputs,
    point_weight: float = 0.5,
    image_weight: float = 0.5,
    unassigned_label: int = -1,
) -> JointSegmentationResult:
    """Fuse point model predictions with projected image predictions."""
    _validate_prediction_shapes(point_prediction, image_prediction)

    point_valid = point_prediction.labels != unassigned_label
    image_valid = image_prediction.labels != unassigned_label
    labels = np.full(point_prediction.labels.shape, unassigned_label, dtype=int)
    source = np.full(point_prediction.labels.shape, "none", dtype="<U10")

    fused_scores = _fuse_scores_if_available(
        point_prediction,
        image_prediction,
        point_weight=point_weight,
        image_weight=image_weight,
        point_valid=point_valid,
        image_valid=image_valid,
    )

    if fused_scores is not None:
        score_valid = point_valid | image_valid
        labels[score_valid] = np.argmax(fused_scores[score_valid], axis=1)
    else:
        labels = _fuse_hard_labels(
            point_prediction.labels,
            image_prediction.labels,
            point_valid=point_valid,
            image_valid=image_valid,
            point_weight=point_weight,
            image_weight=image_weight,
            unassigned_label=unassigned_label,
        )

    source[point_valid & ~image_valid] = "point"
    source[image_valid & ~point_valid] = "image"
    source[point_valid & image_valid] = "both"

    return JointSegmentationResult(
        labels=labels,
        scores=fused_scores,
        source=source,
        point_weight=point_weight,
        image_weight=image_weight,
    )


def summarize_joint_segmentation(
    result: JointSegmentationResult,
    point_prediction: PredictionInputs,
    image_prediction: PredictionInputs,
    unassigned_label: int = -1,
) -> JointSegmentationSummary:
    """Summarize fused joint predictions."""
    point_valid = point_prediction.labels != unassigned_label
    image_valid = image_prediction.labels != unassigned_label
    fused_valid = result.labels != unassigned_label
    unique, counts = np.unique(result.labels[fused_valid], return_counts=True)

    return JointSegmentationSummary(
        total_point_count=len(result.labels),
        fused_point_count=int(fused_valid.sum()),
        point_model_count=int(point_valid.sum()),
        image_projection_count=int(image_valid.sum()),
        both_sources_count=int((point_valid & image_valid).sum()),
        label_counts={int(label): int(count) for label, count in zip(unique, counts)},
    )


def save_joint_segmentation_npz(path: str | Path, result: JointSegmentationResult) -> None:
    """Save fused labels, source tags, and optional scores."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "assigned_labels": result.labels,
        "joint_labels": result.labels,
        "source": result.source,
        "point_weight": result.point_weight,
        "image_weight": result.image_weight,
    }
    if result.scores is not None:
        payload["class_scores"] = result.scores
    np.savez_compressed(output_path, **payload)


def save_joint_summary(path: str | Path, summary: JointSegmentationSummary) -> None:
    """Save joint segmentation summary as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")


def _load_labels(prediction, point_count: int) -> np.ndarray:
    if "assigned_labels" in prediction:
        labels = np.asarray(prediction["assigned_labels"], dtype=int)
    elif "point_labels" in prediction:
        labels = np.asarray(prediction["point_labels"], dtype=int)
    elif {"point_indices", "class_scores"}.issubset(prediction.files):
        labels = np.full((point_count,), -1, dtype=int)
        class_scores = prediction["class_scores"]
        if class_scores.ndim == 1:
            labels[prediction["point_indices"]] = class_scores.astype(int)
        else:
            labels[prediction["point_indices"]] = np.argmax(class_scores, axis=1).astype(int)
    else:
        raise ValueError("Prediction file must contain labels or point_indices/class_scores.")

    if labels.shape != (point_count,):
        raise ValueError("Prediction labels must contain one value per point.")
    return labels


def _load_scores(prediction, point_count: int) -> np.ndarray | None:
    if "class_scores" not in prediction:
        return None

    scores = np.asarray(prediction["class_scores"], dtype=float)
    if scores.ndim != 2:
        return None
    if scores.shape[0] == point_count:
        return scores
    if "point_indices" in prediction:
        full_scores = np.zeros((point_count, scores.shape[1]), dtype=float)
        full_scores[prediction["point_indices"]] = scores
        return full_scores
    raise ValueError("class_scores must be N x C or paired with point_indices.")


def _fuse_scores_if_available(
    point_prediction: PredictionInputs,
    image_prediction: PredictionInputs,
    point_weight: float,
    image_weight: float,
    point_valid: np.ndarray,
    image_valid: np.ndarray,
) -> np.ndarray | None:
    if point_prediction.scores is None or image_prediction.scores is None:
        return None
    if point_prediction.scores.shape != image_prediction.scores.shape:
        raise ValueError("Point and image score arrays must have the same shape for score fusion.")

    fused = np.zeros_like(point_prediction.scores, dtype=float)
    weights = np.zeros((len(point_prediction.labels), 1), dtype=float)

    fused[point_valid] += point_prediction.scores[point_valid] * point_weight
    weights[point_valid] += point_weight
    fused[image_valid] += image_prediction.scores[image_valid] * image_weight
    weights[image_valid] += image_weight

    valid = weights[:, 0] > 0
    fused[valid] = fused[valid] / weights[valid]
    return fused


def _fuse_hard_labels(
    point_labels: np.ndarray,
    image_labels: np.ndarray,
    point_valid: np.ndarray,
    image_valid: np.ndarray,
    point_weight: float,
    image_weight: float,
    unassigned_label: int,
) -> np.ndarray:
    labels = np.full(point_labels.shape, unassigned_label, dtype=int)
    labels[point_valid & ~image_valid] = point_labels[point_valid & ~image_valid]
    labels[image_valid & ~point_valid] = image_labels[image_valid & ~point_valid]

    both = point_valid & image_valid
    agree = both & (point_labels == image_labels)
    labels[agree] = point_labels[agree]

    conflict = both & (point_labels != image_labels)
    prefer_image = image_weight > point_weight
    labels[conflict] = np.where(prefer_image, image_labels[conflict], point_labels[conflict])
    return labels


def _validate_prediction_shapes(
    point_prediction: PredictionInputs,
    image_prediction: PredictionInputs,
) -> None:
    if point_prediction.labels.shape != image_prediction.labels.shape:
        raise ValueError("Point and image predictions must have the same point count.")

