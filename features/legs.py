from __future__ import annotations

from vision.landmarks import Landmarks


class LegFeatures:
    """Extract rhythm descriptors from leg landmarks."""

    ANKLES = [15, 16]

    def __init__(self, landmarks: Landmarks) -> None:
        self._lm = landmarks

    def ankle_velocity(self) -> float:
        """Mean velocity of both ankles."""
        return self._lm.mean_velocity(self.ANKLES)
