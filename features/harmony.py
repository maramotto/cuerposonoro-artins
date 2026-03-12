from __future__ import annotations

from vision.landmarks import Landmarks


class HarmonyFeatures:
    """Extract harmony descriptors from torso and head tilt."""

    def __init__(self, landmarks: Landmarks) -> None:
        self._lm = landmarks

    def torso_tilt(self) -> float:
        """Lateral torso tilt: positive = right, negative = left.

        Computed as the difference between the midpoint of shoulders
        and the midpoint of hips on the x-axis.
        """
        shoulder_mid_x = (
            float(self._lm.position(5)[0]) + float(self._lm.position(6)[0])
        ) / 2.0
        hip_mid_x = (
            float(self._lm.position(11)[0]) + float(self._lm.position(12)[0])
        ) / 2.0
        return shoulder_mid_x - hip_mid_x

    def head_tilt(self) -> float:
        """Lateral head tilt: positive = right, negative = left.

        Computed as the difference in y between right and left ear.
        Right ear lower (higher y) means tilting right → positive.
        """
        left_ear_y = float(self._lm.position(3)[1])
        right_ear_y = float(self._lm.position(4)[1])
        return right_ear_y - left_ear_y

    def should_advance(self, threshold: float) -> bool:
        return self.torso_tilt() > threshold

    def should_retreat(self, threshold: float) -> bool:
        return self.torso_tilt() < -threshold
