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

## iPhone LiDAR Projection

The first projection script maps image-space labels back onto a LiDAR point cloud:

```bash
python scripts/project_iphone_lidar.py \
  --points data/raw/pointclouds/scan.npy \
  --labels data/raw/labels/frame_labels.npy \
  --calibration data/raw/calibration/frame_calibration.json \
  --output outputs/projections/scan_projected_labels.npz
```

Supported point inputs are `.npy`, `.npz`, `.csv`, and ASCII `.ply`. Calibration JSON should include a 3x3 `intrinsics` matrix and may include a 4x4 `camera_from_lidar` matrix. If `camera_from_lidar` is missing, points are treated as already being in the camera coordinate frame.

For a regular photo without iPhone LiDAR calibration, use the relative projection script. It assumes points are already camera-relative unless a `camera_from_world` transform is provided:

```bash
python scripts/project_relative_photo.py \
  --points data/raw/pointclouds/scan.npy \
  --labels data/raw/labels/photo_labels.npy \
  --fov-degrees 60 \
  --output outputs/projections/photo_projected_labels.npz
```

You can also provide `--camera data/raw/calibration/photo_camera.json` with either `intrinsics` or `fov_degrees`, plus an optional 4x4 `camera_from_world` transform.

To visualize either iPhone LiDAR or regular-photo projection output:

```bash
python scripts/visualize_projection.py \
  --points data/raw/pointclouds/scan.npy \
  --projection outputs/projections/scan_projected_labels.npz \
  --output outputs/projections/scan_projected_labels.png
```

For the relative-photo projection, pass the `.npz` from `project_relative_photo.py`:

```bash
python scripts/visualize_projection.py \
  --points data/raw/pointclouds/scan.npy \
  --projection outputs/projections/photo_projected_labels.npz \
  --output outputs/projections/photo_projected_labels.png \
  --title "Projected Photo Labels"
```

Omit `--output` to open an interactive Matplotlib window.

To compare the original point cloud with projected labels side by side:

```bash
python scripts/compare_projection.py \
  --points data/raw/pointclouds/scan.npy \
  --projection outputs/projections/photo_projected_labels.npz \
  --output outputs/projections/photo_projection_comparison.png
```

To classify a point cloud using the projected image labels:

```bash
python scripts/classify_projected_pointcloud.py \
  --points data/raw/pointclouds/scan.npy \
  --projection outputs/projections/photo_projected_labels.npz \
  --output outputs/classifications/scan_classes.npz \
  --summary outputs/classifications/scan_summary.json
```

## Point Model Inference

The first point-model backend is Open3D-ML RandLA-Net. RandLA-Net is a lightweight semantic segmentation architecture for large-scale point clouds that uses random sampling and local feature aggregation.

Install optional model dependencies before running this path:

```bash
pip install -e ".[models]"
```

The model extra pins the compatible Open3D-ML stack used by this project: Open3D `0.19.*`, Torch `2.2.*`, TorchVision `0.17.*`, Dash `2.*`, and TensorBoard.

Then run inference with a RandLA-Net checkpoint:

```bash
python scripts/run_point_model_inference.py \
  --points data/raw/pointclouds/scan.npy \
  --checkpoint checkpoints/randlanet.ckpt \
  --num-classes 19 \
  --output outputs/model_predictions/scan_randlanet.npz
```

The output includes `assigned_labels`, so it can be passed directly to `visualize_projection.py` or `compare_projection.py`.
RandLA-Net expects at least 1024 points after loading; real scans should comfortably exceed this.

To verify dependencies:

```bash
python scripts/run_point_model_inference.py --check-dependencies
```

To use Open3D's pretrained SemanticKITTI checkpoint:

```bash
python scripts/run_point_model_inference.py \
  --points data/raw/pointclouds/scan.npy \
  --download-semantic-kitti-checkpoint \
  --output outputs/model_predictions/scan_randlanet.npz
```
