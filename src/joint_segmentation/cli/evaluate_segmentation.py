"""Evaluate predicted point labels against ground truth labels."""

from __future__ import annotations

import argparse

from joint_segmentation.evaluation.segmentation_metrics import (
    evaluate_segmentation,
    load_labels,
    print_evaluation_report,
    save_evaluation_report,
)
from joint_segmentation.labels import load_label_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, help="Predicted labels: .npz, .npy, or .csv.")
    parser.add_argument("--ground-truth", required=True, help="Ground truth labels: .npz, .npy, or .csv.")
    parser.add_argument("--output", help="Optional JSON report path.")
    parser.add_argument("--point-count", type=int, help="Required for sparse .npz prediction inputs.")
    parser.add_argument("--ignore-label", type=int, default=-1, help="Ground-truth label to ignore.")
    parser.add_argument("--label-map", help="Optional YAML label map for class names in reports.")
    parser.add_argument(
        "--classes",
        nargs="*",
        type=int,
        help="Optional explicit class labels to include in mean IoU.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predicted = load_labels(args.prediction, point_count=args.point_count)
    ground_truth = load_labels(args.ground_truth, point_count=len(predicted))
    label_map = load_label_map(args.label_map)
    evaluation = evaluate_segmentation(
        predicted,
        ground_truth,
        ignore_label=args.ignore_label,
        class_labels=args.classes,
        label_map=label_map,
    )

    print_evaluation_report(evaluation)
    if args.output:
        save_evaluation_report(args.output, evaluation)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
