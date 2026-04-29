import numpy as np

from joint_segmentation.models.open3d_randlanet import (
    Open3DRandLANetSegmenter,
    PointModelPrediction,
    extract_open3d_predictions,
    save_point_model_prediction,
)


class FakePipeline:
    def __init__(self, results):
        self.results = results
        self.last_data = None

    def run_inference(self, data):
        self.last_data = data
        return self.results


def test_extract_open3d_predictions_from_scores() -> None:
    labels, scores = extract_open3d_predictions(
        {"predict_scores": np.array([[0.2, 0.8], [0.9, 0.1]])}
    )

    np.testing.assert_array_equal(labels, np.array([1, 0]))
    np.testing.assert_allclose(scores, np.array([[0.2, 0.8], [0.9, 0.1]]))


def test_open3d_randlanet_segmenter_uses_injected_pipeline() -> None:
    pipeline = FakePipeline({"predict_labels": np.array([2, 3])})
    segmenter = Open3DRandLANetSegmenter(pipeline=pipeline)

    prediction = segmenter.predict(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]))

    np.testing.assert_array_equal(prediction.labels, np.array([2, 3]))
    assert pipeline.last_data["point"].shape == (2, 3)
    assert pipeline.last_data["feat"].shape == (2, 3)


def test_save_point_model_prediction_is_visualizer_compatible(tmp_path) -> None:
    output_path = tmp_path / "prediction.npz"
    prediction = PointModelPrediction(
        labels=np.array([1, 2]),
        scores=np.array([[0.1, 0.9, 0.0], [0.2, 0.3, 0.5]]),
    )

    save_point_model_prediction(output_path, prediction)

    saved = np.load(output_path)
    np.testing.assert_array_equal(saved["assigned_labels"], np.array([1, 2]))
    np.testing.assert_allclose(saved["class_scores"], prediction.scores)

