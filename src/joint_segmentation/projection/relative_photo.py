"""Approximate projection from a regular photo into a point cloud.

This module is for early experiments where there is no device-specific LiDAR
calibration. It assumes points are either already in the photo camera frame or
can be moved there with a user-provided 4x4 `camera_from_world` transform.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from joint_segmentation.projection.iphone_lidar import project_lidar_labels
from joint_segmentation.projection.iphone_lidar import save_projection as save_projection
from joint_segmentation.projection.projector import ProjectionResult


@dataclass(frozen=True)
class RelativePhotoCamera:
    """Camera estimate for projecting world points into a regular photo."""

    intrinsics: np.ndarray
    camera_from_world: np.ndarray

    @classmethod
    def from_json(cls, path: str | Path, image_width: int, image_height: int) -> "RelativePhotoCamera":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_payload(payload, image_width=image_width, image_height=image_height)

    @classmethod
    def from_payload(
        cls,
        payload: dict,
        image_width: int,
        image_height: int,
    ) -> "RelativePhotoCamera":
        if "intrinsics" in payload:
            intrinsics = np.asarray(payload["intrinsics"], dtype=float)
        else:
            fov_degrees = float(payload.get("fov_degrees", 60.0))
            intrinsics = intrinsics_from_horizontal_fov(image_width, image_height, fov_degrees)

        camera_from_world = np.asarray(payload.get("camera_from_world", np.eye(4)), dtype=float)

        if intrinsics.shape != (3, 3):
            raise ValueError("Photo intrinsics must be a 3x3 matrix.")
        if camera_from_world.shape != (4, 4):
            raise ValueError("camera_from_world must be a 4x4 matrix when provided.")

        return cls(intrinsics=intrinsics, camera_from_world=camera_from_world)


def intrinsics_from_horizontal_fov(
    image_width: int,
    image_height: int,
    fov_degrees: float,
) -> np.ndarray:
    """Estimate a pinhole camera matrix from image size and horizontal FOV."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive.")
    if not 0.0 < fov_degrees < 180.0:
        raise ValueError("Horizontal FOV must be between 0 and 180 degrees.")

    focal = (image_width / 2.0) / math.tan(math.radians(fov_degrees) / 2.0)
    return np.array(
        [
            [focal, 0.0, image_width / 2.0],
            [0.0, focal, image_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def project_relative_photo_labels(
    points_world: np.ndarray,
    image_labels: np.ndarray,
    camera: RelativePhotoCamera,
) -> ProjectionResult:
    """Project regular-photo labels onto a point cloud with an approximate camera."""
    from joint_segmentation.projection.iphone_lidar import CameraCalibration

    calibration = CameraCalibration(
        intrinsics=camera.intrinsics,
        camera_from_lidar=camera.camera_from_world,
    )
    return project_lidar_labels(points_world, image_labels, calibration)


def camera_payload_template(image_width: int = 1920, image_height: int = 1080) -> dict:
    """Return a minimal relative-photo camera JSON payload."""
    return {
        "image_width": image_width,
        "image_height": image_height,
        "fov_degrees": 60.0,
        "camera_from_world": np.eye(4).tolist(),
    }

