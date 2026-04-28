"""Visualize projected labels on a point cloud."""

from __future__ import annotations

import argparse

from joint_segmentation.projection.iphone_lidar import load_points
from joint_segmentation.visualization.projection_viewer import (
    load_projection_labels,
    render_projection_plot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", required=True, help="Path to .npy, .npz, .csv, or ASCII .ply points.")
    parser.add_argument("--projection", required=True, help="Path to projection .npz output.")
    parser.add_argument("--output", help="Optional PNG path. If omitted, opens an interactive window.")
    parser.add_argument("--max-points", type=int, default=100_000, help="Maximum points to plot.")
    parser.add_argument("--point-size", type=float, default=1.0, help="Scatter point size.")
    parser.add_argument("--title", default="Projected iPhone LiDAR Labels", help="Plot title.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = load_points(args.points)
    labels = load_projection_labels(args.projection, point_count=len(points))

    render_projection_plot(
        points,
        labels,
        output=args.output,
        max_points=args.max_points,
        point_size=args.point_size,
        title=args.title,
        show=args.output is None,
    )

    if args.output:
        print(f"Wrote visualization to {args.output}")


if __name__ == "__main__":
    main()

