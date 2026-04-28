"""Dataset placeholders for scanned point clouds and image assets."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScanSample:
    """Paths for one scan and its related image/calibration data."""

    point_cloud_path: Path
    image_path: Path | None = None
    calibration_path: Path | None = None
    label_path: Path | None = None


class JointSegmentationDataset:
    """Minimal dataset shell to be replaced with real loading logic."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.samples: list[ScanSample] = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ScanSample:
        return self.samples[index]

