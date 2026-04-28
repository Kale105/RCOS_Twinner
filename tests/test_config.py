from joint_segmentation.config import load_config


def test_load_default_config() -> None:
    config = load_config("configs/default.yaml")

    assert config["project"]["name"] == "joint-pointcloud-segmentation"

