"""End-to-end orchestration for projection, point inference, fusion, and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from joint_segmentation.config import load_config
from joint_segmentation.evaluation.segmentation_metrics import (
    evaluate_segmentation,
    load_labels,
    save_evaluation_report,
)
from joint_segmentation.fusion.joint_segmentation import (
    PredictionInputs,
    fuse_point_and_image_predictions,
    load_prediction_inputs,
    save_joint_segmentation_npz,
    save_joint_summary,
    summarize_joint_segmentation,
)
from joint_segmentation.models.open3d_randlanet import (
    Open3DRandLANetSegmenter,
    download_semantic_kitti_checkpoint,
    save_point_model_prediction,
)
from joint_segmentation.projection.iphone_lidar import (
    CameraCalibration,
    load_image_labels,
    load_points,
    project_lidar_labels,
    save_projection,
)
from joint_segmentation.projection.relative_photo import (
    RelativePhotoCamera,
    intrinsics_from_horizontal_fov,
    project_relative_photo_labels,
)
from joint_segmentation.visualization.projection_viewer import (
    load_projection_labels,
    render_projection_comparison,
    render_projection_plot,
)


@dataclass(frozen=True)
class PipelineOutputs:
    """Paths produced by the end-to-end pipeline."""

    image_projection: Path
    point_prediction: Path
    joint_prediction: Path
    joint_summary: Path
    evaluation_report: Path | None
    visualization: Path | None
    comparison: Path | None


def run_pipeline_from_config(config_path: str | Path) -> PipelineOutputs:
    """Run the end-to-end pipeline from a YAML config file."""
    config = load_config(config_path)
    base_dir = Path(config.get("output_dir", "outputs/pipeline"))
    base_dir.mkdir(parents=True, exist_ok=True)

    points_path = _required_path(config, "points")
    points = load_points(points_path)
    point_count = len(points)

    image_projection_path = Path(config.get("image_projection_output", base_dir / "image_projection.npz"))
    point_prediction_path = Path(config.get("point_prediction_output", base_dir / "point_prediction.npz"))
    joint_prediction_path = Path(config.get("joint_output", base_dir / "joint_labels.npz"))
    joint_summary_path = Path(config.get("joint_summary", base_dir / "joint_summary.json"))

    _run_projection(config, points, image_projection_path)
    _run_point_prediction(config, points, point_prediction_path)

    point_prediction = load_prediction_inputs(point_prediction_path, point_count=point_count)
    image_prediction = load_prediction_inputs(image_projection_path, point_count=point_count)
    fusion_config = config.get("fusion", {})
    joint_result = fuse_point_and_image_predictions(
        point_prediction,
        image_prediction,
        point_weight=float(fusion_config.get("point_weight", 0.5)),
        image_weight=float(fusion_config.get("image_weight", 0.5)),
        unassigned_label=int(fusion_config.get("unassigned_label", -1)),
    )
    save_joint_segmentation_npz(joint_prediction_path, joint_result)
    save_joint_summary(
        joint_summary_path,
        summarize_joint_segmentation(
            joint_result,
            point_prediction,
            image_prediction,
            unassigned_label=int(fusion_config.get("unassigned_label", -1)),
        ),
    )

    evaluation_report = _run_evaluation(config, joint_prediction_path, point_count)
    visualization, comparison = _run_visualizations(config, points, joint_prediction_path, base_dir)

    return PipelineOutputs(
        image_projection=image_projection_path,
        point_prediction=point_prediction_path,
        joint_prediction=joint_prediction_path,
        joint_summary=joint_summary_path,
        evaluation_report=evaluation_report,
        visualization=visualization,
        comparison=comparison,
    )


def _run_projection(config: dict[str, Any], points: np.ndarray, output_path: Path) -> None:
    projection_config = config.get("projection", {})
    projection_type = projection_config.get("type", "relative_photo")
    labels = load_image_labels(_required_path(projection_config, "labels"))

    if projection_type == "iphone_lidar":
        calibration = CameraCalibration.from_json(_required_path(projection_config, "calibration"))
        result = project_lidar_labels(points, labels, calibration)
    elif projection_type == "relative_photo":
        height, width = labels.shape[:2]
        if projection_config.get("camera"):
            camera = RelativePhotoCamera.from_json(
                projection_config["camera"],
                image_width=int(projection_config.get("image_width", width)),
                image_height=int(projection_config.get("image_height", height)),
            )
        else:
            camera = RelativePhotoCamera(
                intrinsics=intrinsics_from_horizontal_fov(
                    int(projection_config.get("image_width", width)),
                    int(projection_config.get("image_height", height)),
                    float(projection_config.get("fov_degrees", 60.0)),
                ),
                camera_from_world=np.eye(4),
            )
        result = project_relative_photo_labels(points, labels, camera)
    else:
        raise ValueError(f"Unsupported projection type: {projection_type}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_projection(output_path, result, point_count=len(points))


def _run_point_prediction(config: dict[str, Any], points: np.ndarray, output_path: Path) -> None:
    model_config = config.get("point_model", {})
    if not bool(model_config.get("run", True)):
        labels = np.full((len(points),), int(model_config.get("fallback_label", -1)), dtype=int)
        scores = _one_hot_scores(labels, int(model_config.get("num_classes", 19)))
        save_point_model_prediction(output_path, PredictionInputs(labels=labels, scores=scores))
        return

    checkpoint = model_config.get("checkpoint")
    if model_config.get("download_semantic_kitti_checkpoint"):
        checkpoint = str(
            download_semantic_kitti_checkpoint(
                Path(model_config.get("checkpoint_dir", "checkpoints"))
                / "randlanet_semantickitti_202201071330utc.pth"
            )
        )

    segmenter = Open3DRandLANetSegmenter(
        checkpoint=checkpoint,
        num_classes=int(model_config.get("num_classes", 19)),
        device=model_config.get("device", "cpu"),
        num_points=int(model_config.get("num_points", 45_056)),
    )
    prediction = segmenter.predict(points)
    save_point_model_prediction(output_path, prediction)


def _run_evaluation(config: dict[str, Any], joint_prediction_path: Path, point_count: int) -> Path | None:
    evaluation_config = config.get("evaluation", {})
    ground_truth_path = evaluation_config.get("ground_truth")
    if not ground_truth_path:
        return None

    output_path = Path(evaluation_config.get("output", Path(config.get("output_dir", "outputs/pipeline")) / "evaluation.json"))
    predicted = load_labels(joint_prediction_path, point_count=point_count)
    ground_truth = load_labels(ground_truth_path, point_count=point_count)
    evaluation = evaluate_segmentation(
        predicted,
        ground_truth,
        ignore_label=int(evaluation_config.get("ignore_label", -1)),
        class_labels=evaluation_config.get("classes"),
    )
    save_evaluation_report(output_path, evaluation)
    return output_path


def _run_visualizations(
    config: dict[str, Any],
    points: np.ndarray,
    joint_prediction_path: Path,
    base_dir: Path,
) -> tuple[Path | None, Path | None]:
    visualization_config = config.get("visualization", {})
    if not visualization_config.get("enabled", False):
        return None, None

    labels = load_projection_labels(joint_prediction_path, point_count=len(points))
    point_size = float(visualization_config.get("point_size", 1.0))
    max_points = visualization_config.get("max_points", 100_000)

    visualization_path = Path(visualization_config.get("output", base_dir / "joint_labels.png"))
    render_projection_plot(
        points,
        labels,
        output=visualization_path,
        max_points=max_points,
        point_size=point_size,
        title=visualization_config.get("title", "Joint Segmentation Labels"),
    )

    comparison_path = None
    if visualization_config.get("comparison", True):
        comparison_path = Path(
            visualization_config.get("comparison_output", base_dir / "joint_comparison.png")
        )
        render_projection_comparison(
            points,
            labels,
            output=comparison_path,
            max_points=max_points,
            point_size=point_size,
            title=visualization_config.get("comparison_title", "Joint Segmentation Comparison"),
        )

    return visualization_path, comparison_path


def _one_hot_scores(labels: np.ndarray, num_classes: int) -> np.ndarray:
    scores = np.zeros((len(labels), num_classes), dtype=float)
    valid = (labels >= 0) & (labels < num_classes)
    scores[np.arange(len(labels))[valid], labels[valid]] = 1.0
    return scores


def _required_path(config: dict[str, Any], key: str) -> Path:
    if key not in config:
        raise ValueError(f"Missing required config key: {key}")
    return Path(config[key])

