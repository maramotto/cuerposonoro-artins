import numpy as np
import pytest
from vision.landmarks import Landmarks
from features.legs import LegFeatures


def _make_landmarks(ankle_y=0.9):
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, 2] = 1.0
    kp[13] = [0.4, 0.7, 1.0]   # left knee
    kp[14] = [0.6, 0.7, 1.0]   # right knee
    kp[15] = [0.4, ankle_y, 1.0]  # left ankle
    kp[16] = [0.6, ankle_y, 1.0]  # right ankle
    return kp


class TestLegFeatures:
    def test_ankle_velocity_first_frame(self):
        lm = Landmarks(_make_landmarks())
        feat = LegFeatures(lm)
        assert feat.ankle_velocity() == pytest.approx(0.0)

    def test_ankle_velocity_after_step(self):
        kp1 = _make_landmarks()
        kp2 = _make_landmarks()
        kp2[15, 0] += 0.2  # left ankle moves
        kp2[16, 0] += 0.2  # right ankle moves

        lm = Landmarks(kp1)
        lm.update(kp2)
        feat = LegFeatures(lm)
        assert feat.ankle_velocity() > 0.1
