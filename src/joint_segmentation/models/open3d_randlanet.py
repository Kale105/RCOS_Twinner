"""Open3D-ML RandLA-Net inference adapter.

RandLA-Net is kept as an optional backend because Open3D-ML and Torch are heavy
runtime dependencies. Import errors are raised only when this adapter is used.
"""

from __future__ import annotations

import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SEMANTIC_KITTI_RANDLANET_URL = (
    "https://storage.googleapis.com/open3d-releases/model-zoo/"
    "randlanet_semantickitti_202201071330utc.pth"
)


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
        if len(points) < 1024:
            raise ValueError(
                "RandLA-Net inference needs at least 1024 input points for its "
                "multi-layer neighbor sampling. Use a larger scan or a simpler baseline."
            )

        data = {
            "point": points,
            "feat": None,
            "label": np.zeros((len(points),), dtype=np.int32),
        }
        results = self.pipeline.run_inference(data)
        labels, scores = extract_open3d_predictions(results)
        return PointModelPrediction(labels=labels, scores=scores)

    def _build_pipeline(self) -> Any:
        if self.backend != "torch":
            raise ValueError("Only the Open3D-ML Torch backend is wired for this adapter.")

        configure_open3d_runtime()
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


def check_open3d_randlanet_dependencies() -> dict[str, str]:
    """Import the Open3D-ML RandLA-Net stack and return installed versions."""
    configure_open3d_runtime()

    import dash
    import open3d
    import tensorboard
    import torch
    import torchvision
    from open3d.ml.torch.models import RandLANet
    from open3d.ml.torch.pipelines import SemanticSegmentation

    # Keep imported symbols referenced so static checkers do not treat this as dead code.
    _ = (RandLANet, SemanticSegmentation)
    return {
        "dash": dash.__version__,
        "open3d": open3d.__version__,
        "tensorboard": tensorboard.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }


def download_semantic_kitti_checkpoint(
    output_path: str | Path,
    url: str = SEMANTIC_KITTI_RANDLANET_URL,
) -> Path:
    """Download the Open3D model-zoo RandLA-Net SemanticKITTI checkpoint."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path

    urllib.request.urlretrieve(url, output_path)
    return output_path


def configure_open3d_runtime() -> None:
    """Set writable cache directories before Open3D imports visualization modules."""
    cache_root = Path(tempfile.gettempdir()) / "joint_segmentation_open3d"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")


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
