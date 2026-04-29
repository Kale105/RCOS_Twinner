"""Metrics for per-point segmentation predictions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from joint_segmentation.fusion.joint_segmentation import load_prediction_inputs
from joint_segmentation.labels import LabelMap, label_map_to_jsonable


@dataclass(frozen=True)
class ClassMetric:
    """Per-class segmentation metrics."""

    label: int
    name: str | None
    true_positive: int
    false_positive: int
    false_negative: int
    support: int
    predicted: int
    iou: float | None
    precision: float | None
    recall: float | None


@dataclass(frozen=True)
class SegmentationEvaluation:
    """Segmentation evaluation summary."""

    total_point_count: int
    evaluated_point_count: int
    ignored_point_count: int
    accuracy: float | None
    mean_iou: float | None
    class_metrics: list[ClassMetric]
    label_map: dict | None = None


def load_labels(path: str | Path, point_count: int | None = None, preferred_key: str | None = None) -> np.ndarray:
    """Load per-point labels from `.npy`, `.npz`, or `.csv`."""
    label_path = Path(path)
    suffix = label_path.suffix.lower()

    if suffix == ".npy":
        labels = np.load(label_path)
    elif suffix == ".npz":
        if point_count is None:
            point_count = _infer_point_count_from_npz(label_path, preferred_key)
        labels = load_prediction_inputs(label_path, point_count=point_count).labels
    elif suffix == ".csv":
        labels = np.loadtxt(label_path, delimiter=",", dtype=int)
    else:
        raise ValueError(f"Unsupported label format: {suffix}")

    labels = np.asarray(labels, dtype=int).reshape((-1,))
    if point_count is not None and labels.shape != (point_count,):
        raise ValueError("Loaded labels do not match expected point count.")
    return labels


def evaluate_segmentation(
    predicted_labels: np.ndarray,
    ground_truth_labels: np.ndarray,
    ignore_label: int = -1,
    class_labels: list[int] | None = None,
    label_map: LabelMap | None = None,
) -> SegmentationEvaluation:
    """Evaluate per-point segmentation labels against ground truth."""
    predicted = np.asarray(predicted_labels, dtype=int).reshape((-1,))
    ground_truth = np.asarray(ground_truth_labels, dtype=int).reshape((-1,))
    if predicted.shape != ground_truth.shape:
        raise ValueError("Predicted and ground-truth labels must have the same shape.")

    valid = ground_truth != ignore_label
    evaluated_count = int(valid.sum())
    ignored_count = int((~valid).sum())

    if class_labels is None:
        class_values = sorted(
            int(label)
            for label in np.unique(np.concatenate([predicted[valid], ground_truth[valid]]))
            if label != ignore_label
        )
    else:
        class_values = [int(label) for label in class_labels]

    accuracy = None
    if evaluated_count > 0:
        accuracy = float((predicted[valid] == ground_truth[valid]).sum() / evaluated_count)

    class_metrics = [
        _class_metric(label, predicted=predicted, ground_truth=ground_truth, valid=valid, label_map=label_map)
        for label in class_values
    ]
    ious = [metric.iou for metric in class_metrics if metric.iou is not None]
    mean_iou = float(np.mean(ious)) if ious else None

    return SegmentationEvaluation(
        total_point_count=len(predicted),
        evaluated_point_count=evaluated_count,
        ignored_point_count=ignored_count,
        accuracy=accuracy,
        mean_iou=mean_iou,
        class_metrics=class_metrics,
        label_map=label_map_to_jsonable(label_map),
    )


def save_evaluation_report(path: str | Path, evaluation: SegmentationEvaluation) -> None:
    """Save evaluation metrics as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(evaluation), indent=2), encoding="utf-8")


def print_evaluation_report(evaluation: SegmentationEvaluation) -> None:
    """Print a concise evaluation report."""
    accuracy = _format_optional(evaluation.accuracy)
    mean_iou = _format_optional(evaluation.mean_iou)
    print(f"Evaluated points: {evaluation.evaluated_point_count} / {evaluation.total_point_count}")
    print(f"Ignored points: {evaluation.ignored_point_count}")
    print(f"Accuracy: {accuracy}")
    print(f"Mean IoU: {mean_iou}")


def _class_metric(
    label: int,
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    valid: np.ndarray,
    label_map: LabelMap | None = None,
) -> ClassMetric:
    pred_is_label = (predicted == label) & valid
    truth_is_label = (ground_truth == label) & valid
    true_positive = int((pred_is_label & truth_is_label).sum())
    false_positive = int((pred_is_label & ~truth_is_label).sum())
    false_negative = int((~pred_is_label & truth_is_label).sum())
    support = int(truth_is_label.sum())
    predicted_count = int(pred_is_label.sum())

    union = true_positive + false_positive + false_negative
    iou = None if union == 0 else true_positive / union
    precision = None if predicted_count == 0 else true_positive / predicted_count
    recall = None if support == 0 else true_positive / support

    return ClassMetric(
        label=label,
        name=None if label_map is None else label_map.name(label),
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        support=support,
        predicted=predicted_count,
        iou=None if iou is None else float(iou),
        precision=None if precision is None else float(precision),
        recall=None if recall is None else float(recall),
    )


def _infer_point_count_from_npz(path: Path, preferred_key: str | None) -> int:
    prediction = np.load(path)
    keys = [preferred_key] if preferred_key else []
    keys.extend(["assigned_labels", "joint_labels", "point_labels"])
    for key in keys:
        if key and key in prediction:
            return int(np.asarray(prediction[key]).reshape((-1,)).shape[0])
    raise ValueError("point_count is required for sparse .npz labels.")


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"
