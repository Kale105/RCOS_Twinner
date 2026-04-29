"""Classify point clouds using projected image labels."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from joint_segmentation.visualization.projection_viewer import load_projection_labels


@dataclass(frozen=True)
class PointCloudClassification:
    """Classification output derived from projected image labels."""

    point_labels: np.ndarray
    projected_mask: np.ndarray
    dominant_label: int | None
    projected_point_count: int
    total_point_count: int
    label_counts: dict[int, int]

    @property
    def projected_fraction(self) -> float:
        if self.total_point_count == 0:
            return 0.0
        return self.projected_point_count / self.total_point_count


def classify_from_projection(
    projection_path: str | Path,
    point_count: int,
    unprojected_label: int = -1,
) -> PointCloudClassification:
    """Assign point labels from a projection file and summarize the result."""
    point_labels = load_projection_labels(projection_path, point_count=point_count)
    projected_mask = point_labels != unprojected_label
    projected_labels = point_labels[projected_mask]

    label_counts = count_labels(projected_labels)
    dominant_label = max(label_counts, key=label_counts.get) if label_counts else None

    return PointCloudClassification(
        point_labels=point_labels,
        projected_mask=projected_mask,
        dominant_label=dominant_label,
        projected_point_count=int(projected_mask.sum()),
        total_point_count=int(point_count),
        label_counts=label_counts,
    )


def count_labels(labels: np.ndarray) -> dict[int, int]:
    """Count integer labels."""
    if len(labels) == 0:
        return {}

    unique, counts = np.unique(labels.astype(int), return_counts=True)
    return {int(label): int(count) for label, count in zip(unique, counts)}


def save_classification_npz(path: str | Path, classification: PointCloudClassification) -> None:
    """Save per-point classification outputs."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        point_labels=classification.point_labels,
        projected_mask=classification.projected_mask,
        dominant_label=-1 if classification.dominant_label is None else classification.dominant_label,
        projected_point_count=classification.projected_point_count,
        total_point_count=classification.total_point_count,
    )


def save_classification_summary(path: str | Path, classification: PointCloudClassification) -> None:
    """Save a JSON summary of the projected-label classification."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **asdict(classification),
        "point_labels": None,
        "projected_mask": None,
        "projected_fraction": classification.projected_fraction,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

