"""Classify a point cloud using projected image labels."""

from __future__ import annotations

import argparse

from joint_segmentation.classification.projected_image import (
    classify_from_projection,
    save_classification_npz,
    save_classification_summary,
)
from joint_segmentation.projection.iphone_lidar import load_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", required=True, help="Path to .npy, .npz, .csv, or ASCII .ply points.")
    parser.add_argument("--projection", required=True, help="Path to projection .npz output.")
    parser.add_argument("--output", required=True, help="Output .npz path for per-point classes.")
    parser.add_argument("--summary", help="Optional output JSON path for class counts and dominant label.")
    parser.add_argument(
        "--unprojected-label",
        type=int,
        default=-1,
        help="Point label value used for points with no projected image label.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = load_points(args.points)
    classification = classify_from_projection(
        args.projection,
        point_count=len(points),
        unprojected_label=args.unprojected_label,
    )

    save_classification_npz(args.output, classification)
    if args.summary:
        save_classification_summary(args.summary, classification)

    label_text = "none" if classification.dominant_label is None else str(classification.dominant_label)
    print(
        "Classified "
        f"{classification.projected_point_count} / {classification.total_point_count} points "
        f"from projected image labels. Dominant label: {label_text}."
    )
    print(f"Wrote {args.output}")
    if args.summary:
        print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()

