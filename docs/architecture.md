# Architecture Notes

## Goal

Build a joint point cloud segmentation pipeline that combines direct 3D point segmentation with image-derived predictions projected into the point cloud.

## Main Components

- `data`: dataset loaders and calibration handling
- `models`: point segmentation model wrappers
- `projection`: image-to-point projection utilities
- `fusion`: label fusion strategies
- `training`: training and evaluation loops

## Open Questions

- Which point segmentation backbone should be used first?
- What image segmentation model will provide projected labels?
- What calibration format will be standardized?
- Should fusion happen at logits, probabilities, or discrete labels?

