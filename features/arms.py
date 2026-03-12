from __future__ import annotations

from vision.landmarks import Landmarks, COCO_ARM


class ArmFeatures:
    """Extract melody descriptors from arm landmarks."""

    WRISTS = [9, 10]

    def __init__(self, landmarks: Landmarks) -> None:
        self._lm = landmarks

    def mean_wrist_height(self) -> float:
        """Mean normalised height of both wrists (0=bottom, 1=top)."""
        left = self._lm.height(9)
        right = self._lm.height(10)
        return (left + right) / 2.0

    def wrist_separation(self) -> float:
        """Horizontal distance between wrists (normalised, 0-1)."""
        left_x = float(self._lm.position(9)[0])
        right_x = float(self._lm.position(10)[0])
        return abs(right_x - left_x)

    def arm_velocity(self) -> float:
        """Mean velocity across all arm landmarks."""
        return self._lm.mean_velocity(COCO_ARM)

    def brightness(self) -> int:
        """Map wrist separation to MIDI CC74 brightness (0-127)."""
        separation = self.wrist_separation()
        return int(min(127, max(0, separation * 127)))
