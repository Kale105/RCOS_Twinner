import json

import numpy as np

from joint_segmentation.evaluation.segmentation_metrics import (
    evaluate_segmentation,
    load_labels,
    save_evaluation_report,
)
from joint_segmentation.labels import LabelInfo, LabelMap


def test_evaluate_segmentation_computes_accuracy_and_iou() -> None:
    predicted = np.array([1, 1, 2, 2, 3])
    ground_truth = np.array([1, 2, 2, -1, 3])

    evaluation = evaluate_segmentation(predicted, ground_truth)

    assert evaluation.total_point_count == 5
    assert evaluation.evaluated_point_count == 4
    assert evaluation.ignored_point_count == 1
    assert evaluation.accuracy == 0.75
    assert evaluation.mean_iou == (0.5 + 0.5 + 1.0) / 3


def test_load_labels_reads_joint_npz(tmp_path) -> None:
    labels_path = tmp_path / "joint.npz"
    np.savez(labels_path, assigned_labels=np.array([4, 5, -1]))

    labels = load_labels(labels_path)

    np.testing.assert_array_equal(labels, np.array([4, 5, -1]))


def test_load_labels_expands_sparse_prediction_npz(tmp_path) -> None:
    labels_path = tmp_path / "sparse.npz"
    np.savez(labels_path, point_indices=np.array([0, 2]), class_scores=np.array([7, 8]))

    labels = load_labels(labels_path, point_count=3)

    np.testing.assert_array_equal(labels, np.array([7, -1, 8]))


def test_save_evaluation_report(tmp_path) -> None:
    output_path = tmp_path / "report.json"
    evaluation = evaluate_segmentation(np.array([1, 2]), np.array([1, 1]))

    save_evaluation_report(output_path, evaluation)

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["accuracy"] == 0.5
    assert report["class_metrics"][0]["label"] == 1


def test_evaluate_segmentation_includes_label_names() -> None:
    label_map = LabelMap({1: LabelInfo(id=1, name="car", color="#2457ff")})

    evaluation = evaluate_segmentation(np.array([1]), np.array([1]), label_map=label_map)

    assert evaluation.class_metrics[0].name == "car"
    assert evaluation.label_map["1"]["name"] == "car"
