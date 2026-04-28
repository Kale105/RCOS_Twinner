"""Project iPhone LiDAR points into an image label map."""

from __future__ import annotations

import argparse
from pathlib import Path

from joint_segmentation.projection.iphone_lidar import (
    CameraCalibration,
    load_image_labels,
    load_points,
    project_lidar_labels,
    save_projection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", required=True, help="Path to .npy, .npz, .csv, or ASCII .ply points.")
    parser.add_argument("--labels", required=True, help="Path to .npy/.npz image labels or scores.")
    parser.add_argument("--calibration", required=True, help="Path to camera calibration JSON.")
    parser.add_argument("--output", required=True, help="Output .npz path for projected labels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = load_points(args.points)
    labels = load_image_labels(args.labels)
    calibration = CameraCalibration.from_json(args.calibration)

    result = project_lidar_labels(points, labels, calibration)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_projection(output_path, result, point_count=len(points))

    print(f"Projected labels for {len(result.point_indices)} / {len(points)} points.")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
