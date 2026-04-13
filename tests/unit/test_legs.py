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
        """First frame has no previous data, velocity is zero."""
        lm = Landmarks(_make_landmarks())
        feat = LegFeatures(lm)
        assert feat.ankle_velocity() == pytest.approx(0.0)

    def test_ankle_velocity_after_step(self):
        """Walking (both ankles move) produces velocity above threshold."""
        kp1 = _make_landmarks()
        kp2 = _make_landmarks()
        kp2[15, 0] += 0.2  # left ankle moves
        kp2[16, 0] += 0.2  # right ankle moves

        lm = Landmarks(kp1)
        lm.update(kp2)
        feat = LegFeatures(lm)
        assert feat.ankle_velocity() > 0.1

    def test_single_ankle_movement(self):
        """Only one ankle moving produces half the mean velocity."""
        kp1 = _make_landmarks()
        kp2 = _make_landmarks()
        kp2[15, 0] += 0.2  # only left ankle moves

        lm = Landmarks(kp1)
        lm.update(kp2)
        feat = LegFeatures(lm)
        vel = feat.ankle_velocity()
        # Mean of (0.2, 0.0) = 0.1
        assert vel == pytest.approx(0.1, abs=0.01)

    def test_vertical_ankle_movement(self):
        """Vertical movement (lifting feet) also produces velocity."""
        kp1 = _make_landmarks()
        kp2 = _make_landmarks()
        kp2[15, 1] -= 0.15  # left ankle moves up
        kp2[16, 1] -= 0.15  # right ankle moves up

        lm = Landmarks(kp1)
        lm.update(kp2)
        feat = LegFeatures(lm)
        assert feat.ankle_velocity() > 0.1

    def test_tiny_jitter_below_threshold(self):
        """Sub-pixel jitter produces velocity well below typical trigger."""
        kp1 = _make_landmarks()
        kp2 = _make_landmarks()
        kp2[15, 0] += 0.002
        kp2[16, 0] -= 0.002

        lm = Landmarks(kp1)
        lm.update(kp2)
        feat = LegFeatures(lm)
        assert feat.ankle_velocity() < 0.01

    def test_uses_correct_landmark_indices(self):
        """LegFeatures.ANKLES should reference COCO ankles (15, 16)."""
        assert LegFeatures.ANKLES == [15, 16]

    def test_velocity_scales_with_movement(self):
        """Faster movement produces proportionally higher velocity."""
        kp1 = _make_landmarks()

        kp_slow = _make_landmarks()
        kp_slow[15, 0] += 0.05
        kp_slow[16, 0] += 0.05

        kp_fast = _make_landmarks()
        kp_fast[15, 0] += 0.20
        kp_fast[16, 0] += 0.20

        lm_slow = Landmarks(kp1.copy())
        lm_slow.update(kp_slow)

        lm_fast = Landmarks(kp1.copy())
        lm_fast.update(kp_fast)

        vel_slow = LegFeatures(lm_slow).ankle_velocity()
        vel_fast = LegFeatures(lm_fast).ankle_velocity()

        assert vel_fast > vel_slow
        assert vel_fast == pytest.approx(vel_slow * 4, abs=0.01)
