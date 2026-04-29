import yaml
import numpy as np

from joint_segmentation.pipeline.end_to_end import run_pipeline_from_config


def test_run_pipeline_from_config_without_point_model(tmp_path) -> None:
    points_path = tmp_path / "points.npy"
    labels_path = tmp_path / "photo_labels.npy"
    truth_path = tmp_path / "truth.npy"
    config_path = tmp_path / "pipeline.yaml"
    output_dir = tmp_path / "outputs"

    np.save(points_path, np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [3.0, 0.0, 2.0]]))
    np.save(labels_path, np.array([[0, 1, 2, 3], [4, 5, 6, 7]]))
    np.save(truth_path, np.array([6, 7, -1]))
    config_path.write_text(
        yaml.safe_dump(
            {
                "points": str(points_path),
                "output_dir": str(output_dir),
                "projection": {
                    "type": "relative_photo",
                    "labels": str(labels_path),
                    "fov_degrees": 90,
                },
                "point_model": {
                    "run": False,
                    "fallback_label": -1,
                    "num_classes": 8,
                },
                "fusion": {
                    "point_weight": 0.0,
                    "image_weight": 1.0,
                },
                "evaluation": {
                    "ground_truth": str(truth_path),
                    "output": str(output_dir / "evaluation.json"),
                },
                "visualization": {
                    "enabled": False,
                },
            }
        ),
        encoding="utf-8",
    )

    outputs = run_pipeline_from_config(config_path)

    assert outputs.image_projection.exists()
    assert outputs.point_prediction.exists()
    assert outputs.joint_prediction.exists()
    assert outputs.joint_summary.exists()
    assert outputs.evaluation_report is not None
    assert outputs.evaluation_report.exists()
