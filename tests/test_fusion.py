import numpy as np

from joint_segmentation.fusion.weighted_vote import fuse_scores


def test_fuse_scores_uses_weights() -> None:
    point_scores = np.array([[1.0, 0.0]])
    projected_scores = np.array([[0.0, 1.0]])

    fused = fuse_scores(point_scores, projected_scores, point_weight=0.75, image_weight=0.25)

    np.testing.assert_allclose(fused, np.array([[0.75, 0.25]]))

