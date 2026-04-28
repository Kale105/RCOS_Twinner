# Joint Point Cloud Segmentation

Initial scaffold for a joint segmentation model that labels scanned point clouds by combining:

- an AI point cloud segmentation model
- image-guided labels projected back into 3D point space

This repository is intentionally light on implementation for the initial commit. The goal is to establish project structure, interfaces, configuration, and documentation placeholders before model code is added.

## Project Layout

```text
configs/                 Experiment and data configuration files
data/                    Local dataset mount points and notes
docs/                    Design notes and implementation plans
notebooks/               Exploration notebooks
scripts/                 Small command-line helpers
src/joint_segmentation/  Python package source
tests/                   Unit and smoke tests
```

## Planned Pipeline

1. Load scanned point clouds and calibrated image assets.
2. Run point cloud segmentation on the scan.
3. Project image-space predictions into the point cloud.
4. Fuse point-model and projected-image labels.
5. Export per-point semantic labels and evaluation metrics.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

