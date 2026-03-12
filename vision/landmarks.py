from __future__ import annotations

import numpy as np

# COCO 17 landmark groups
COCO_ARM = [5, 6, 7, 8, 9, 10]
COCO_LEG = [13, 14, 15, 16]
COCO_EAR = [3, 4]
COCO_TORSO = [5, 6, 11, 12]


class Landmarks:
    """Normalised COCO-17 landmarks with velocity tracking."""

    def __init__(self, keypoints: np.ndarray) -> None:
        self.keypoints = keypoints  # shape (17, 3): x, y, confidence
        self._prev: np.ndarray | None = None

    def update(self, keypoints: np.ndarray) -> None:
        self._prev = self.keypoints.copy()
        self.keypoints = keypoints

    def position(self, index: int) -> np.ndarray:
        return self.keypoints[index, :2]

    def positions(self, indices: list[int]) -> np.ndarray:
        return self.keypoints[indices, :2]

    def mean_position(self, indices: list[int]) -> np.ndarray:
        return self.positions(indices).mean(axis=0)

    def confidence(self, index: int) -> float:
        return float(self.keypoints[index, 2])

    def height(self, index: int) -> float:
        """Return normalised height (1 = top, 0 = bottom). Inverts image y."""
        return 1.0 - float(self.keypoints[index, 1])

    def velocity(self, index: int) -> float:
        """Euclidean displacement of a landmark since last update."""
        if self._prev is None:
            return 0.0
        delta = self.keypoints[index, :2] - self._prev[index, :2]
        return float(np.linalg.norm(delta))

    def mean_velocity(self, indices: list[int]) -> float:
        return float(np.mean([self.velocity(i) for i in indices]))
