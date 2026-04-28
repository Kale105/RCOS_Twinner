"""Project regular-photo labels onto a point cloud with approximate camera geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from joint_segmentation.projection.iphone_lidar import load_image_labels, load_points
from joint_segmentation.projection.relative_photo import (
    RelativePhotoCamera,
    intrinsics_from_horizontal_fov,
    project_relative_photo_labels,
    save_projection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", required=True, help="Path to .npy, .npz, .csv, or ASCII .ply points.")
    parser.add_argument("--labels", required=True, help="Path to .npy/.npz photo labels or scores.")
    parser.add_argument("--output", required=True, help="Output .npz path for projected labels.")
    parser.add_argument("--camera", help="Optional JSON with intrinsics/fov and camera_from_world.")
    parser.add_argument("--image-width", type=int, help="Photo width in pixels when no camera JSON is used.")
    parser.add_argument("--image-height", type=int, help="Photo height in pixels when no camera JSON is used.")
    parser.add_argument(
        "--fov-degrees",
        type=float,
        default=60.0,
        help="Estimated horizontal field of view when no intrinsics are provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = load_points(args.points)
    labels = load_image_labels(args.labels)
    image_height, image_width = labels.shape[:2]

    if args.camera:
        camera = RelativePhotoCamera.from_json(
            args.camera,
            image_width=args.image_width or image_width,
            image_height=args.image_height or image_height,
        )
    else:
        width = args.image_width or image_width
        height = args.image_height or image_height
        camera = RelativePhotoCamera(
            intrinsics=intrinsics_from_horizontal_fov(width, height, args.fov_degrees),
            camera_from_world=np.eye(4),
        )

    result = project_relative_photo_labels(points, labels, camera)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_projection(output_path, result, point_count=len(points))

    print(f"Projected photo labels for {len(result.point_indices)} / {len(points)} points.")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
