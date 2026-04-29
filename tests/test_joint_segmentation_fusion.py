import json

import numpy as np

from joint_segmentation.fusion.joint_segmentation import (
    PredictionInputs,
    fuse_point_and_image_predictions,
    load_prediction_inputs,
    save_joint_segmentation_npz,
    save_joint_summary,
    summarize_joint_segmentation,
)


def test_fuse_hard_labels_prefers_higher_weight_on_conflict() -> None:
    point_prediction = PredictionInputs(labels=np.array([1, 2, -1]), scores=None)
    image_prediction = PredictionInputs(labels=np.array([1, 3, 4]), scores=None)

    result = fuse_point_and_image_predictions(
        point_prediction,
        image_prediction,
        point_weight=0.25,
        image_weight=0.75,
    )

    np.testing.assert_array_equal(result.labels, np.array([1, 3, 4]))
    np.testing.assert_array_equal(result.source, np.array(["both", "both", "image"]))


def test_fuse_scores_when_both_predictions_have_scores() -> None:
    point_prediction = PredictionInputs(
        labels=np.array([0, 1]),
        scores=np.array([[0.8, 0.2], [0.3, 0.7]]),
    )
    image_prediction = PredictionInputs(
        labels=np.array([1, 1]),
        scores=np.array([[0.1, 0.9], [0.4, 0.6]]),
    )

    result = fuse_point_and_image_predictions(point_prediction, image_prediction)

    np.testing.assert_array_equal(result.labels, np.array([1, 1]))
    np.testing.assert_allclose(result.scores, np.array([[0.45, 0.55], [0.35, 0.65]]))


def test_load_prediction_inputs_expands_projected_scores(tmp_path) -> None:
    prediction_path = tmp_path / "projection.npz"
    np.savez(
        prediction_path,
        point_indices=np.array([0, 2]),
        class_scores=np.array([[0.1, 0.9], [0.8, 0.2]]),
    )

    prediction = load_prediction_inputs(prediction_path, point_count=3)

    np.testing.assert_array_equal(prediction.labels, np.array([1, -1, 0]))
    np.testing.assert_allclose(prediction.scores, np.array([[0.1, 0.9], [0.0, 0.0], [0.8, 0.2]]))


def test_save_joint_outputs(tmp_path) -> None:
    output_path = tmp_path / "joint.npz"
    summary_path = tmp_path / "joint.json"
    point_prediction = PredictionInputs(labels=np.array([1, 2]), scores=None)
    image_prediction = PredictionInputs(labels=np.array([1, -1]), scores=None)
    result = fuse_point_and_image_predictions(point_prediction, image_prediction)
    summary = summarize_joint_segmentation(result, point_prediction, image_prediction)

    save_joint_segmentation_npz(output_path, result)
    save_joint_summary(summary_path, summary)

    saved = np.load(output_path)
    np.testing.assert_array_equal(saved["assigned_labels"], np.array([1, 2]))
    assert json.loads(summary_path.read_text(encoding="utf-8"))["both_sources_count"] == 1
