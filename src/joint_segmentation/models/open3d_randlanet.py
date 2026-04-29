"""Open3D-ML RandLA-Net inference adapter.

RandLA-Net is kept as an optional backend because Open3D-ML and Torch are heavy
runtime dependencies. Import errors are raised only when this adapter is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PointModelPrediction:
    """Per-point semantic segmentation prediction."""

    labels: np.ndarray
    scores: np.ndarray | None = None


class Open3DRandLANetSegmenter:
    """Run Open3D-ML RandLA-Net semantic segmentation inference."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        num_classes: int = 19,
        device: str = "cpu",
        num_points: int = 45_056,
        backend: str = "torch",
        pipeline: Any | None = None,
    ) -> None:
        self.checkpoint = None if checkpoint is None else str(checkpoint)
        self.num_classes = num_classes
        self.device = device
        self.num_points = num_points
        self.backend = backend
        self.pipeline = pipeline or self._build_pipeline()

    def predict(self, points: np.ndarray) -> PointModelPrediction:
        """Predict per-point semantic labels for an N x 3 point cloud."""
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("RandLA-Net inference expects points shaped N x 3.")

        data = {
            "point": points,
            "feat": points,
            "label": np.zeros((len(points),), dtype=np.int32),
        }
        results = self.pipeline.run_inference(data)
        labels, scores = extract_open3d_predictions(results)
        return PointModelPrediction(labels=labels, scores=scores)

    def _build_pipeline(self) -> Any:
        if self.backend != "torch":
            raise ValueError("Only the Open3D-ML Torch backend is wired for this adapter.")

        try:
            from open3d.ml.torch.models import RandLANet
            from open3d.ml.torch.pipelines import SemanticSegmentation
        except ImportError as exc:
            raise ImportError(
                "Open3D-ML RandLA-Net inference requires Open3D with ML support and Torch. "
                "Install the optional model dependencies before using this backend."
            ) from exc

        model = RandLANet(
            ckpt_path=self.checkpoint,
            num_classes=self.num_classes,
            num_points=self.num_points,
            in_channels=3,
        )
        pipeline = SemanticSegmentation(
            model,
            dataset=None,
            device=self.device,
            test_batch_size=1,
        )
        if self.checkpoint:
            pipeline.load_ckpt(self.checkpoint)
        return pipeline


def extract_open3d_predictions(results: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Normalize common Open3D-ML inference result shapes into labels and scores."""
    if isinstance(results, dict):
        labels = _first_present_or_none(results, ("predict_labels", "labels", "pred", "predict"))
        scores = _first_present_or_none(results, ("predict_scores", "scores", "logits", "probs"))
        if labels is None and scores is None:
            raise ValueError("Could not find labels or scores in Open3D-ML inference results.")
        if labels is None:
            labels = scores
    else:
        labels = results
        scores = None

    labels_array = np.asarray(labels)
    scores_array = None if scores is None else np.asarray(scores)

    if labels_array.ndim == 2:
        scores_array = labels_array
        labels_array = np.argmax(labels_array, axis=1)
    elif labels_array.ndim == 3 and labels_array.shape[0] == 1:
        scores_array = labels_array[0]
        labels_array = np.argmax(scores_array, axis=1)

    if labels_array.ndim != 1:
        raise ValueError("Could not extract one semantic label per point from model results.")

    return labels_array.astype(int), scores_array


def save_point_model_prediction(path: str | Path, prediction: PointModelPrediction) -> None:
    """Save point model predictions in an `.npz` format compatible with visualizers."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "point_labels": prediction.labels,
        "assigned_labels": prediction.labels,
    }
    if prediction.scores is not None:
        payload["class_scores"] = prediction.scores
    np.savez_compressed(output_path, **payload)


def _first_present_or_none(results: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in results:
            return results[key]
    return None
