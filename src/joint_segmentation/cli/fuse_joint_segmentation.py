"""Fuse point-model segmentation with projected image labels."""

from __future__ import annotations

import argparse

from joint_segmentation.fusion.joint_segmentation import (
    fuse_point_and_image_predictions,
    load_prediction_inputs,
    save_joint_segmentation_npz,
    save_joint_summary,
    summarize_joint_segmentation,
)
from joint_segmentation.projection.iphone_lidar import load_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", required=True, help="Path to .npy, .npz, .csv, or ASCII .ply points.")
    parser.add_argument("--point-prediction", required=True, help="Point model prediction .npz.")
    parser.add_argument("--image-projection", required=True, help="Camera projection prediction .npz.")
    parser.add_argument("--output", required=True, help="Output .npz path for fused labels.")
    parser.add_argument("--summary", help="Optional JSON summary path.")
    parser.add_argument("--point-weight", type=float, default=0.5, help="Weight for point model output.")
    parser.add_argument("--image-weight", type=float, default=0.5, help="Weight for image projection output.")
    parser.add_argument("--unassigned-label", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    point_count = len(load_points(args.points))
    point_prediction = load_prediction_inputs(args.point_prediction, point_count=point_count)
    image_prediction = load_prediction_inputs(args.image_projection, point_count=point_count)

    result = fuse_point_and_image_predictions(
        point_prediction,
        image_prediction,
        point_weight=args.point_weight,
        image_weight=args.image_weight,
        unassigned_label=args.unassigned_label,
    )
    save_joint_segmentation_npz(args.output, result)

    if args.summary:
        summary = summarize_joint_segmentation(
            result,
            point_prediction,
            image_prediction,
            unassigned_label=args.unassigned_label,
        )
        save_joint_summary(args.summary, summary)

    print(f"Fused joint segmentation for {len(result.labels)} points.")
    print(f"Wrote {args.output}")
    if args.summary:
        print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()

