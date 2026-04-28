"""Visualize projected image labels on a 3D point cloud."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ProjectionVisualizationData:
    """Prepared point cloud data for visualization."""

    points: np.ndarray
    labels: np.ndarray


def load_projection_labels(path: str | Path, point_count: int) -> np.ndarray:
    """Load one label per point from projection output."""
    projection = np.load(path)

    if "assigned_labels" in projection:
        labels = np.asarray(projection["assigned_labels"], dtype=int)
    elif {"point_indices", "class_scores"}.issubset(projection.files):
        labels = np.full((point_count,), -1, dtype=int)
        class_scores = projection["class_scores"]
        if class_scores.ndim == 1:
            labels[projection["point_indices"]] = class_scores.astype(int)
        else:
            labels[projection["point_indices"]] = np.argmax(class_scores, axis=1).astype(int)
    else:
        raise ValueError("Projection file must contain assigned_labels or point_indices/class_scores.")

    if labels.shape != (point_count,):
        raise ValueError("Projection labels must have one value per point.")

    return labels


def prepare_visualization_data(
    points: np.ndarray,
    labels: np.ndarray,
    max_points: int | None = 100_000,
) -> ProjectionVisualizationData:
    """Validate and optionally downsample point cloud visualization data."""
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels, dtype=int)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be shaped N x 3.")
    if labels.shape != (len(points),):
        raise ValueError("labels must contain one value per point.")
    if max_points is not None and max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, num=max_points, dtype=int)
        points = points[indices]
        labels = labels[indices]

    return ProjectionVisualizationData(points=points, labels=labels)


def render_projection_plot(
    points: np.ndarray,
    labels: np.ndarray,
    output: str | Path | None = None,
    max_points: int | None = 100_000,
    point_size: float = 1.0,
    title: str = "Projected iPhone LiDAR Labels",
    show: bool = False,
) -> None:
    """Render a 3D scatter plot of projected labels."""
    _configure_matplotlib_cache()

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    data = prepare_visualization_data(points, labels, max_points=max_points)
    color_values, tick_values, tick_labels = label_color_values(data.labels)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        data.points[:, 0],
        data.points[:, 1],
        data.points[:, 2],
        c=color_values,
        cmap=ListedColormap(_label_palette(len(tick_values))),
        s=point_size,
        linewidths=0,
    )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect(_axis_aspect(data.points))

    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.08)
    colorbar.set_ticks(tick_values)
    colorbar.set_ticklabels(tick_labels)
    colorbar.set_label("Projected label")
    fig.tight_layout()

    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def label_color_values(labels: np.ndarray) -> tuple[np.ndarray, list[int], list[str]]:
    """Map raw labels into compact color ids for plotting."""
    unique_labels = sorted(int(label) for label in np.unique(labels))
    label_to_color = {label: index for index, label in enumerate(unique_labels)}
    color_values = np.array([label_to_color[int(label)] for label in labels], dtype=int)
    tick_values = list(range(len(unique_labels)))
    tick_labels = ["unprojected" if label == -1 else str(label) for label in unique_labels]
    return color_values, tick_values, tick_labels


def _label_palette(size: int) -> list[str]:
    base = [
        "#8a8f98",
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#17becf",
        "#bcbd22",
    ]
    if size <= len(base):
        return base[:size]
    return [base[index % len(base)] for index in range(size)]


def _axis_aspect(points: np.ndarray) -> tuple[float, float, float]:
    ranges = np.ptp(points, axis=0)
    ranges = np.where(ranges == 0, 1.0, ranges)
    return tuple(float(value) for value in ranges)


def _configure_matplotlib_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "joint_segmentation_matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
