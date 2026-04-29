"""Run point cloud segmentation model inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from joint_segmentation.models.open3d_randlanet import (
    Open3DRandLANetSegmenter,
    SEMANTIC_KITTI_RANDLANET_URL,
    check_open3d_randlanet_dependencies,
    download_semantic_kitti_checkpoint,
    save_point_model_prediction,
)
from joint_segmentation.projection.iphone_lidar import load_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", help="Path to .npy, .npz, .csv, or ASCII .ply points.")
    parser.add_argument("--output", help="Output .npz path for model predictions.")
    parser.add_argument("--checkpoint", help="Path to a RandLA-Net checkpoint.")
    parser.add_argument(
        "--download-semantic-kitti-checkpoint",
        action="store_true",
        help="Download Open3D's pretrained SemanticKITTI RandLA-Net checkpoint if needed.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Directory used with --download-semantic-kitti-checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-url",
        default=SEMANTIC_KITTI_RANDLANET_URL,
        help="Checkpoint URL used with --download-semantic-kitti-checkpoint.",
    )
    parser.add_argument("--num-classes", type=int, default=19, help="Number of semantic classes.")
    parser.add_argument("--num-points", type=int, default=45_056, help="RandLA-Net sampled point count.")
    parser.add_argument("--device", default="cpu", help="Open3D-ML device, such as cpu or cuda.")
    parser.add_argument("--backend", default="torch", choices=["torch"], help="Open3D-ML backend.")
    parser.add_argument(
        "--check-dependencies",
        action="store_true",
        help="Import Open3D-ML/Torch dependencies and print installed versions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_dependencies:
        versions = check_open3d_randlanet_dependencies()
        print("Open3D-ML RandLA-Net dependencies are available:")
        for package, version in versions.items():
            print(f"  {package}: {version}")
        return

    if not args.points or not args.output:
        raise SystemExit("--points and --output are required unless --check-dependencies is used.")

    checkpoint = args.checkpoint
    if args.download_semantic_kitti_checkpoint:
        checkpoint_path = Path(args.checkpoint_dir) / "randlanet_semantickitti_202201071330utc.pth"
        checkpoint = str(download_semantic_kitti_checkpoint(checkpoint_path, url=args.checkpoint_url))

    points = load_points(args.points)
    segmenter = Open3DRandLANetSegmenter(
        checkpoint=checkpoint,
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
