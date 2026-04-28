"""Projection helpers for iPhone LiDAR scans and camera-frame labels.

The first supported calibration format is a JSON file with:

```json
{
  "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "camera_from_lidar": [[...], [...], [...], [...]]
}
```

If `camera_from_lidar` is omitted, points are assumed to already be in the
camera coordinate frame. This matches a common early workflow when ARKit export
code has already transformed LiDAR points before saving them.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from joint_segmentation.projection.projector import ProjectionResult


@dataclass(frozen=True)
class CameraCalibration:
    """Camera calibration needed to project 3D points into image pixels."""

    intrinsics: np.ndarray
    camera_from_lidar: np.ndarray

    @classmethod
    def from_json(cls, path: str | Path) -> "CameraCalibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        intrinsics = np.asarray(payload["intrinsics"], dtype=float)
        camera_from_lidar = np.asarray(payload.get("camera_from_lidar", np.eye(4)), dtype=float)

        if intrinsics.shape != (3, 3):
            raise ValueError("Calibration intrinsics must be a 3x3 matrix.")
        if camera_from_lidar.shape != (4, 4):
            raise ValueError("camera_from_lidar must be a 4x4 matrix when provided.")

        return cls(intrinsics=intrinsics, camera_from_lidar=camera_from_lidar)


def load_points(path: str | Path) -> np.ndarray:
    """Load point coordinates from `.npy`, `.npz`, `.csv`, or ASCII `.ply`."""
    point_path = Path(path)
    suffix = point_path.suffix.lower()

    if suffix == ".npy":
        points = np.load(point_path)
    elif suffix == ".npz":
        archive = np.load(point_path)
        key = "points" if "points" in archive else archive.files[0]
        points = archive[key]
    elif suffix == ".csv":
        points = _load_csv_points(point_path)
    elif suffix == ".ply":
        points = _load_ascii_ply_points(point_path)
    else:
        raise ValueError(f"Unsupported point cloud format: {suffix}")

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Point cloud must be shaped as N x 3 or wider.")
    return points[:, :3]


def load_image_labels(path: str | Path) -> np.ndarray:
    """Load image-space labels or class scores from a NumPy file."""
    label_path = Path(path)
    if label_path.suffix.lower() not in {".npy", ".npz"}:
        raise ValueError("Projected labels currently support .npy or .npz files.")

    if label_path.suffix.lower() == ".npy":
        return np.load(label_path)

    archive = np.load(label_path)
    key = "labels" if "labels" in archive else archive.files[0]
    return archive[key]


def project_lidar_labels(
    points_lidar: np.ndarray,
    image_labels: np.ndarray,
    calibration: CameraCalibration,
) -> ProjectionResult:
    """Project image labels or scores onto LiDAR points.

    `image_labels` may be a 2D integer label map shaped `H x W` or a 3D score
    tensor shaped `H x W x C`.
    """
    points_lidar = np.asarray(points_lidar, dtype=float)
    labels = np.asarray(image_labels)

    if points_lidar.ndim != 2 or points_lidar.shape[1] != 3:
        raise ValueError("points_lidar must be shaped N x 3.")
    if labels.ndim not in {2, 3}:
        raise ValueError("image_labels must be H x W labels or H x W x C scores.")

    points_camera = transform_points(points_lidar, calibration.camera_from_lidar)
    uv, in_front = project_points(points_camera, calibration.intrinsics)

    height, width = labels.shape[:2]
    rounded_uv = np.rint(uv).astype(int)
    in_bounds = (
        (rounded_uv[:, 0] >= 0)
        & (rounded_uv[:, 0] < width)
        & (rounded_uv[:, 1] >= 0)
        & (rounded_uv[:, 1] < height)
    )
    valid_mask = in_front & in_bounds
    point_indices = np.flatnonzero(valid_mask)
    sampled = labels[rounded_uv[valid_mask, 1], rounded_uv[valid_mask, 0]]

    return ProjectionResult(
        point_indices=point_indices,
        class_scores=sampled,
        uv=rounded_uv,
        valid_mask=valid_mask,
    )


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to N x 3 points."""
    ones = np.ones((points.shape[0], 1), dtype=float)
    homogeneous = np.concatenate([points, ones], axis=1)
    transformed = homogeneous @ transform.T
    return transformed[:, :3]


def project_points(points_camera: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame points to pixel coordinates."""
    z = points_camera[:, 2]
    in_front = z > 0
    safe_z = np.where(in_front, z, 1.0)

    x_norm = points_camera[:, 0] / safe_z
    y_norm = points_camera[:, 1] / safe_z
    u = (intrinsics[0, 0] * x_norm) + intrinsics[0, 2]
    v = (intrinsics[1, 1] * y_norm) + intrinsics[1, 2]
    return np.stack([u, v], axis=1), in_front


def save_projection(path: str | Path, result: ProjectionResult, point_count: int) -> None:
    """Save projection output as an `.npz` file."""
    assigned_labels = np.full((point_count,), -1, dtype=int)
    if result.class_scores.ndim == 1:
        assigned_labels[result.point_indices] = result.class_scores.astype(int)
    elif result.class_scores.ndim == 2:
        assigned_labels[result.point_indices] = np.argmax(result.class_scores, axis=1).astype(int)

    np.savez_compressed(
        path,
        point_indices=result.point_indices,
        class_scores=result.class_scores,
        assigned_labels=assigned_labels,
        uv=result.uv,
        valid_mask=result.valid_mask,
    )


def _load_csv_points(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        sample = csv_file.read(2048)
        csv_file.seek(0)
        has_header = csv.Sniffer().has_header(sample)
        if has_header:
            reader = csv.DictReader(csv_file)
            rows = [[row["x"], row["y"], row["z"]] for row in reader]
        else:
            reader = csv.reader(csv_file)
            rows = [row[:3] for row in reader if row]
    return np.asarray(rows, dtype=float)


def _load_ascii_ply_points(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as ply_file:
        vertex_count = None

        for line in ply_file:
            if line.startswith("format") and "ascii" not in line:
                raise ValueError("Only ASCII PLY point clouds are supported for now.")
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line.strip() == "end_header":
                break

        if vertex_count is None:
            raise ValueError("PLY header is missing `element vertex`.")

        rows: list[list[float]] = []
        for _ in range(vertex_count):
            parts = ply_file.readline().split()
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])

    return np.asarray(rows, dtype=float)


def calibration_template() -> dict[str, Any]:
    """Return an example calibration payload for documentation and tests."""
    return {
        "intrinsics": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 720.0], [0.0, 0.0, 1.0]],
        "camera_from_lidar": np.eye(4).tolist(),
    }
