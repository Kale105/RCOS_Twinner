import yaml

from joint_segmentation.labels import LabelMap, label_map_to_jsonable


def test_label_map_loads_names_and_colors(tmp_path) -> None:
    label_map_path = tmp_path / "labels.yaml"
    label_map_path.write_text(
        yaml.safe_dump(
            {
                "labels": {
                    1: {"name": "car", "color": "#2457ff"},
                    2: "road",
                }
            }
        ),
        encoding="utf-8",
    )

    label_map = LabelMap.from_yaml(label_map_path)

    assert label_map.name(1) == "car"
    assert label_map.color(1) == "#2457ff"
    assert label_map.display_name(2) == "2: road"
    assert label_map.name(99) == "99"
    assert label_map_to_jsonable(label_map)["1"]["name"] == "car"
