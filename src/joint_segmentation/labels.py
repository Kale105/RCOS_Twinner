"""Label map loading and formatting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LabelInfo:
    """Human-readable metadata for a segmentation label."""

    id: int
    name: str
    color: str | None = None


class LabelMap:
    """Lookup table for label names and colors."""

    def __init__(self, labels: dict[int, LabelInfo]) -> None:
        self.labels = labels

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LabelMap":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        labels_payload = payload.get("labels", payload)
        labels: dict[int, LabelInfo] = {}

        for raw_id, value in labels_payload.items():
            label_id = int(raw_id)
            if isinstance(value, str):
                labels[label_id] = LabelInfo(id=label_id, name=value)
            else:
                labels[label_id] = LabelInfo(
                    id=label_id,
                    name=str(value.get("name", label_id)),
                    color=value.get("color"),
                )

        return cls(labels)

    def name(self, label: int) -> str:
        if label == -1:
            return "unprojected"
        return self.labels.get(int(label), LabelInfo(id=int(label), name=str(label))).name

    def display_name(self, label: int) -> str:
        if label == -1:
            return "unprojected"
        return f"{label}: {self.name(label)}"

    def color(self, label: int) -> str | None:
        if label == -1:
            return "#8a8f98"
        info = self.labels.get(int(label))
        return None if info is None else info.color

    def names_by_id(self) -> dict[int, str]:
        return {label_id: info.name for label_id, info in self.labels.items()}


def load_label_map(path: str | Path | None) -> LabelMap | None:
    """Load a label map when a path is provided."""
    if path is None:
        return None
    return LabelMap.from_yaml(path)


def label_map_to_jsonable(label_map: LabelMap | None) -> dict[str, Any] | None:
    """Convert a label map into JSON-friendly metadata."""
    if label_map is None:
        return None
    return {
        str(label_id): {
            "name": info.name,
            "color": info.color,
        }
        for label_id, info in label_map.labels.items()
    }

