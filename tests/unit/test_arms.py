import numpy as np
import pytest
from vision.landmarks import Landmarks
from features.arms import ArmFeatures


def _make_landmarks(wrist_y=0.5, wrist_x_left=0.1, wrist_x_right=0.9):
    """Helper: create landmarks with configurable wrist positions."""
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, 2] = 1.0
    kp[5] = [0.3, 0.3, 1.0]   # left shoulder
    kp[6] = [0.7, 0.3, 1.0]   # right shoulder
    kp[7] = [0.2, 0.4, 1.0]   # left elbow
    kp[8] = [0.8, 0.4, 1.0]   # right elbow
    kp[9] = [wrist_x_left, wrist_y, 1.0]   # left wrist
    kp[10] = [wrist_x_right, wrist_y, 1.0]  # right wrist
    return kp


class TestArmFeatures:
    def test_mean_wrist_height_arms_down(self):
        lm = Landmarks(_make_landmarks(wrist_y=0.9))
        feat = ArmFeatures(lm)
        # y=0.9 → height=0.1
        assert feat.mean_wrist_height() == pytest.approx(0.1)

    def test_mean_wrist_height_arms_up(self):
        lm = Landmarks(_make_landmarks(wrist_y=0.1))
        feat = ArmFeatures(lm)
        # y=0.1 → height=0.9
        assert feat.mean_wrist_height() == pytest.approx(0.9)

    def test_wrist_separation_wide(self):
        lm = Landmarks(_make_landmarks(wrist_x_left=0.1, wrist_x_right=0.9))
        feat = ArmFeatures(lm)
        assert feat.wrist_separation() == pytest.approx(0.8)

    def test_wrist_separation_narrow(self):
        lm = Landmarks(_make_landmarks(wrist_x_left=0.45, wrist_x_right=0.55))
        feat = ArmFeatures(lm)
        assert feat.wrist_separation() == pytest.approx(0.1)

    def test_arm_velocity_first_frame(self):
        lm = Landmarks(_make_landmarks())
        feat = ArmFeatures(lm)
        assert feat.arm_velocity() == pytest.approx(0.0)

    def test_arm_velocity_after_movement(self):
        kp1 = _make_landmarks()
        kp2 = _make_landmarks()
        # Move all arm landmarks right by 0.1
        for i in [5, 6, 7, 8, 9, 10]:
            kp2[i, 0] += 0.1

        lm = Landmarks(kp1)
        lm.update(kp2)
        feat = ArmFeatures(lm)
        assert feat.arm_velocity() > 0.05

    def test_brightness_from_separation(self):
        lm = Landmarks(_make_landmarks(wrist_x_left=0.0, wrist_x_right=1.0))
        feat = ArmFeatures(lm)
        # Max separation → brightness near 127
        assert feat.brightness() >= 120

    def test_brightness_from_narrow(self):
        lm = Landmarks(_make_landmarks(wrist_x_left=0.5, wrist_x_right=0.5))
        feat = ArmFeatures(lm)
        # Zero separation → brightness 0
        assert feat.brightness() == 0
