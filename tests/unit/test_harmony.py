import numpy as np
import pytest
from vision.landmarks import Landmarks
from features.harmony import HarmonyFeatures


def _make_landmarks(
    left_shoulder_x=0.3, right_shoulder_x=0.7,
    left_hip_x=0.4, right_hip_x=0.6,
    left_ear_y=0.1, right_ear_y=0.1,
):
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, 2] = 1.0
    kp[3] = [0.4, left_ear_y, 1.0]
    kp[4] = [0.6, right_ear_y, 1.0]
    kp[5] = [left_shoulder_x, 0.3, 1.0]
    kp[6] = [right_shoulder_x, 0.3, 1.0]
    kp[11] = [left_hip_x, 0.6, 1.0]
    kp[12] = [right_hip_x, 0.6, 1.0]
    return kp


class TestHarmonyFeatures:
    def test_torso_tilt_neutral(self):
        lm = Landmarks(_make_landmarks())
        feat = HarmonyFeatures(lm)
        assert feat.torso_tilt() == pytest.approx(0.0, abs=0.01)

    def test_torso_tilt_right(self):
        # Shoulders shifted right relative to hips
        lm = Landmarks(_make_landmarks(
            left_shoulder_x=0.5, right_shoulder_x=0.9,
        ))
        feat = HarmonyFeatures(lm)
        assert feat.torso_tilt() > 0.05

    def test_torso_tilt_left(self):
        # Shoulders shifted left relative to hips
        lm = Landmarks(_make_landmarks(
            left_shoulder_x=0.1, right_shoulder_x=0.5,
        ))
        feat = HarmonyFeatures(lm)
        assert feat.torso_tilt() < -0.05

    def test_head_tilt_neutral(self):
        lm = Landmarks(_make_landmarks())
        feat = HarmonyFeatures(lm)
        assert feat.head_tilt() == pytest.approx(0.0, abs=0.01)

    def test_head_tilt_right(self):
        # Right ear lower than left → tilting right
        lm = Landmarks(_make_landmarks(left_ear_y=0.08, right_ear_y=0.15))
        feat = HarmonyFeatures(lm)
        assert feat.head_tilt() > 0.02

    def test_head_tilt_left(self):
        # Left ear lower than right → tilting left
        lm = Landmarks(_make_landmarks(left_ear_y=0.15, right_ear_y=0.08))
        feat = HarmonyFeatures(lm)
        assert feat.head_tilt() < -0.02

    def test_should_advance_with_strong_tilt(self):
        lm = Landmarks(_make_landmarks(
            left_shoulder_x=0.5, right_shoulder_x=0.9,
        ))
        feat = HarmonyFeatures(lm)
        assert feat.should_advance(threshold=0.05) is True

    def test_should_retreat_with_left_tilt(self):
        lm = Landmarks(_make_landmarks(
            left_shoulder_x=0.1, right_shoulder_x=0.5,
        ))
        feat = HarmonyFeatures(lm)
        assert feat.should_retreat(threshold=0.05) is True

    def test_no_advance_when_neutral(self):
        lm = Landmarks(_make_landmarks())
        feat = HarmonyFeatures(lm)
        assert feat.should_advance(threshold=0.05) is False
        assert feat.should_retreat(threshold=0.05) is False
