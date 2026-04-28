import json

import numpy as np

from joint_segmentation.projection.iphone_lidar import (
    CameraCalibration,
    load_points,
    project_lidar_labels,
)


def test_project_lidar_labels_samples_label_map() -> None:
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    labels = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ]
    )
    calibration = CameraCalibration(
        intrinsics=np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]),
        camera_from_lidar=np.eye(4),
    )

    result = project_lidar_labels(points, labels, calibration)

    np.testing.assert_array_equal(result.point_indices, np.array([0, 1]))
    np.testing.assert_array_equal(result.class_scores, np.array([4, 5]))
    np.testing.assert_array_equal(result.valid_mask, np.array([True, True, False]))


def test_load_points_reads_csv_with_header(tmp_path) -> None:
    points_path = tmp_path / "points.csv"
    points_path.write_text("x,y,z,intensity\n1,2,3,0.1\n4,5,6,0.2\n", encoding="utf-8")

    points = load_points(points_path)

    np.testing.assert_allclose(points, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


def test_calibration_loads_optional_transform(tmp_path) -> None:
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps({"intrinsics": [[1, 0, 2], [0, 1, 3], [0, 0, 1]]}),
        encoding="utf-8",
    )

    calibration = CameraCalibration.from_json(calibration_path)

    np.testing.assert_allclose(calibration.camera_from_lidar, np.eye(4))

