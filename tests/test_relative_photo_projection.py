import json

import numpy as np

from joint_segmentation.projection.relative_photo import (
    RelativePhotoCamera,
    intrinsics_from_horizontal_fov,
    project_relative_photo_labels,
    save_projection,
)
from joint_segmentation.visualization.projection_viewer import load_projection_labels


def test_intrinsics_from_horizontal_fov_centers_camera() -> None:
    intrinsics = intrinsics_from_horizontal_fov(4, 2, 90.0)

    np.testing.assert_allclose(intrinsics, np.array([[2.0, 0.0, 2.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]))


def test_project_relative_photo_labels_uses_fov_camera() -> None:
    points = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [3.0, 0.0, 2.0]])
    labels = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    camera = RelativePhotoCamera(
        intrinsics=intrinsics_from_horizontal_fov(4, 2, 90.0),
        camera_from_world=np.eye(4),
    )

    result = project_relative_photo_labels(points, labels, camera)

    np.testing.assert_array_equal(result.point_indices, np.array([0, 1]))
    np.testing.assert_array_equal(result.class_scores, np.array([6, 7]))


def test_relative_photo_camera_loads_from_fov_json(tmp_path) -> None:
    camera_path = tmp_path / "camera.json"
    camera_path.write_text(json.dumps({"fov_degrees": 90.0}), encoding="utf-8")

    camera = RelativePhotoCamera.from_json(camera_path, image_width=4, image_height=2)

    np.testing.assert_allclose(camera.intrinsics[0, 0], 2.0)
    np.testing.assert_allclose(camera.camera_from_world, np.eye(4))


def test_relative_photo_projection_can_feed_visualizer(tmp_path) -> None:
    projection_path = tmp_path / "relative_projection.npz"
    camera = RelativePhotoCamera(
        intrinsics=intrinsics_from_horizontal_fov(4, 2, 90.0),
        camera_from_world=np.eye(4),
    )
    result = project_relative_photo_labels(
        np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]]),
        np.array([[0, 1, 2, 3], [4, 5, 6, 7]]),
        camera,
    )

    save_projection(projection_path, result, point_count=2)
    labels = load_projection_labels(projection_path, point_count=2)

    np.testing.assert_array_equal(labels, np.array([6, 7]))
