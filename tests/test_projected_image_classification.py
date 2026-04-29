import json

import numpy as np

from joint_segmentation.classification.projected_image import (
    classify_from_projection,
    count_labels,
    save_classification_npz,
    save_classification_summary,
)


def test_count_labels_counts_integer_labels() -> None:
    assert count_labels(np.array([2, 2, 5, 5, 5])) == {2: 2, 5: 3}


def test_classify_from_projection_summarizes_dominant_label(tmp_path) -> None:
    projection_path = tmp_path / "projection.npz"
    np.savez(projection_path, assigned_labels=np.array([3, -1, 3, 5]))

    classification = classify_from_projection(projection_path, point_count=4)

    np.testing.assert_array_equal(classification.point_labels, np.array([3, -1, 3, 5]))
    np.testing.assert_array_equal(classification.projected_mask, np.array([True, False, True, True]))
    assert classification.dominant_label == 3
    assert classification.projected_point_count == 3
    assert classification.total_point_count == 4
    assert classification.label_counts == {3: 2, 5: 1}


def test_save_classification_outputs(tmp_path) -> None:
    projection_path = tmp_path / "projection.npz"
    output_path = tmp_path / "classification.npz"
    summary_path = tmp_path / "summary.json"
    np.savez(projection_path, assigned_labels=np.array([1, 1, -1]))
    classification = classify_from_projection(projection_path, point_count=3)

    save_classification_npz(output_path, classification)
    save_classification_summary(summary_path, classification)

    saved = np.load(output_path)
    np.testing.assert_array_equal(saved["point_labels"], np.array([1, 1, -1]))
    assert int(saved["dominant_label"]) == 1

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["dominant_label"] == 1
    assert summary["projected_fraction"] == 2 / 3

