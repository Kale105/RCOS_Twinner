import numpy as np

from joint_segmentation.visualization.projection_viewer import (
    label_color_values,
    load_projection_labels,
    prepare_visualization_data,
    render_projection_comparison,
)


def test_label_color_values_names_unprojected_points() -> None:
    color_values, tick_values, tick_labels = label_color_values(np.array([-1, 2, 2, 5]))

    np.testing.assert_array_equal(color_values, np.array([0, 1, 1, 2]))
    assert tick_values == [0, 1, 2]
    assert tick_labels == ["unprojected", "2", "5"]


def test_load_projection_labels_from_scores(tmp_path) -> None:
    projection_path = tmp_path / "projection.npz"
    np.savez(
        projection_path,
        point_indices=np.array([0, 2]),
        class_scores=np.array([[0.1, 0.9], [0.8, 0.2]]),
    )

    labels = load_projection_labels(projection_path, point_count=3)

    np.testing.assert_array_equal(labels, np.array([1, -1, 0]))


def test_prepare_visualization_data_downsamples_evenly() -> None:
    points = np.arange(30, dtype=float).reshape(10, 3)
    labels = np.arange(10)

    data = prepare_visualization_data(points, labels, max_points=4)

    assert data.points.shape == (4, 3)
    np.testing.assert_array_equal(data.labels, np.array([0, 3, 6, 9]))


def test_render_projection_comparison_writes_png(tmp_path) -> None:
    output_path = tmp_path / "comparison.png"
    points = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 2.0]])
    labels = np.array([-1, 2, 3])

    render_projection_comparison(points, labels, output=output_path, max_points=None)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
