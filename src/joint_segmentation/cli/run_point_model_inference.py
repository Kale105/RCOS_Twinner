"""Run point cloud segmentation model inference."""

from __future__ import annotations

import argparse

from joint_segmentation.models.open3d_randlanet import (
    Open3DRandLANetSegmenter,
    save_point_model_prediction,
)
from joint_segmentation.projection.iphone_lidar import load_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", required=True, help="Path to .npy, .npz, .csv, or ASCII .ply points.")
    parser.add_argument("--output", required=True, help="Output .npz path for model predictions.")
    parser.add_argument("--checkpoint", help="Path to a RandLA-Net checkpoint.")
    parser.add_argument("--num-classes", type=int, default=19, help="Number of semantic classes.")
    parser.add_argument("--num-points", type=int, default=45_056, help="RandLA-Net sampled point count.")
    parser.add_argument("--device", default="cpu", help="Open3D-ML device, such as cpu or cuda.")
    parser.add_argument("--backend", default="torch", choices=["torch"], help="Open3D-ML backend.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = load_points(args.points)
    segmenter = Open3DRandLANetSegmenter(
        checkpoint=args.checkpoint,
        num_classes=args.num_classes,
        device=args.device,
        num_points=args.num_points,
        backend=args.backend,
    )
    prediction = segmenter.predict(points)
    save_point_model_prediction(args.output, prediction)

    print(f"Segmented {len(prediction.labels)} points with Open3D-ML RandLA-Net.")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

