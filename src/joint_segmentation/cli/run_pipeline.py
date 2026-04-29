"""Run the end-to-end joint segmentation pipeline from a YAML config."""

from __future__ import annotations

import argparse

from joint_segmentation.pipeline.end_to_end import run_pipeline_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline_from_config(args.config)

    print("Pipeline complete.")
    print(f"Image projection: {outputs.image_projection}")
    print(f"Point prediction: {outputs.point_prediction}")
    print(f"Joint prediction: {outputs.joint_prediction}")
    print(f"Joint summary: {outputs.joint_summary}")
    if outputs.evaluation_report:
        print(f"Evaluation: {outputs.evaluation_report}")
    if outputs.visualization:
        print(f"Visualization: {outputs.visualization}")
    if outputs.comparison:
        print(f"Comparison: {outputs.comparison}")


if __name__ == "__main__":
    main()

