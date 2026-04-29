import numpy as np

from joint_segmentation.models.open3d_randlanet import (
    Open3DRandLANetSegmenter,
    PointModelPrediction,
    configure_open3d_runtime,
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

    points = np.tile(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]), (512, 1))
    pipeline.results = {"predict_labels": np.arange(len(points)) % 4}

    prediction = segmenter.predict(points)

    np.testing.assert_array_equal(prediction.labels, np.arange(len(points)) % 4)
    assert pipeline.last_data["point"].shape == (1024, 3)
    assert pipeline.last_data["feat"] is None


def test_open3d_randlanet_segmenter_rejects_tiny_clouds() -> None:
    segmenter = Open3DRandLANetSegmenter(pipeline=FakePipeline({"predict_labels": np.array([])}))

    try:
        segmenter.predict(np.zeros((12, 3)))
    except ValueError as exc:
        assert "at least 1024" in str(exc)
    else:
        raise AssertionError("Expected tiny point clouds to be rejected.")


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


def test_configure_open3d_runtime_sets_writable_cache_dirs(monkeypatch) -> None:
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    configure_open3d_runtime()

    assert "joint_segmentation_open3d" in __import__("os").environ["MPLCONFIGDIR"]
    assert "joint_segmentation_open3d" in __import__("os").environ["XDG_CACHE_HOME"]
    assert __import__("os").environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"
    assert __import__("os").environ["OMP_NUM_THREADS"] == "1"
