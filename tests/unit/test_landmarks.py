import numpy as np
import pytest
from vision.landmarks import Landmarks, COCO_ARM, COCO_LEG, COCO_EAR, COCO_TORSO


class TestLandmarkConstants:
    def test_arm_indices(self):
        assert COCO_ARM == [5, 6, 7, 8, 9, 10]

    def test_leg_indices(self):
        assert COCO_LEG == [13, 14, 15, 16]

    def test_ear_indices(self):
        assert COCO_EAR == [3, 4]

    def test_torso_indices(self):
        assert COCO_TORSO == [5, 6, 11, 12]


class TestLandmarks:
    def _make_keypoints(self):
        """Create a 17x3 array (x, y, confidence) with known positions."""
        kp = np.zeros((17, 3), dtype=np.float32)
        # Set confidence to 1.0 for all
        kp[:, 2] = 1.0
        # Ears at y=0.1
        kp[3] = [0.4, 0.1, 1.0]  # left ear
        kp[4] = [0.6, 0.1, 1.0]  # right ear
        # Shoulders at y=0.3
        kp[5] = [0.3, 0.3, 1.0]  # left shoulder
        kp[6] = [0.7, 0.3, 1.0]  # right shoulder
        # Elbows at y=0.4
        kp[7] = [0.2, 0.4, 1.0]  # left elbow
        kp[8] = [0.8, 0.4, 1.0]  # right elbow
        # Wrists at y=0.5
        kp[9] = [0.1, 0.5, 1.0]  # left wrist
        kp[10] = [0.9, 0.5, 1.0]  # right wrist
        # Hips at y=0.6
        kp[11] = [0.4, 0.6, 1.0]  # left hip
        kp[12] = [0.6, 0.6, 1.0]  # right hip
        # Knees at y=0.7
        kp[13] = [0.4, 0.7, 1.0]  # left knee
        kp[14] = [0.6, 0.7, 1.0]  # right knee
        # Ankles at y=0.9
        kp[15] = [0.4, 0.9, 1.0]  # left ankle
        kp[16] = [0.6, 0.9, 1.0]  # right ankle
        return kp

    def test_from_yolo_keypoints(self):
        kp = self._make_keypoints()
        lm = Landmarks(kp)
        assert lm.keypoints.shape == (17, 3)

    def test_position_returns_xy(self):
        kp = self._make_keypoints()
        lm = Landmarks(kp)
        pos = lm.position(9)  # left wrist
        np.testing.assert_array_almost_equal(pos, [0.1, 0.5])

    def test_positions_group(self):
        kp = self._make_keypoints()
        lm = Landmarks(kp)
        positions = lm.positions(COCO_EAR)
        assert positions.shape == (2, 2)

    def test_mean_position(self):
        kp = self._make_keypoints()
        lm = Landmarks(kp)
        mean = lm.mean_position([9, 10])  # wrists
        np.testing.assert_array_almost_equal(mean, [0.5, 0.5])

    def test_confidence(self):
        kp = self._make_keypoints()
        kp[9, 2] = 0.3
        lm = Landmarks(kp)
        assert lm.confidence(9) == pytest.approx(0.3)

    def test_velocity_first_frame_is_zero(self):
        kp = self._make_keypoints()
        lm = Landmarks(kp)
        vel = lm.velocity(9)
        assert vel == pytest.approx(0.0)

    def test_velocity_after_update(self):
        kp1 = self._make_keypoints()
        kp2 = self._make_keypoints()
        kp2[9] = [0.2, 0.5, 1.0]  # left wrist moved right by 0.1

        lm = Landmarks(kp1)
        lm.update(kp2)
        vel = lm.velocity(9)
        assert vel == pytest.approx(0.1, abs=1e-5)

    def test_mean_velocity(self):
        kp1 = self._make_keypoints()
        kp2 = self._make_keypoints()
        # Move both wrists by the same amount
        kp2[9] = [0.2, 0.5, 1.0]   # +0.1 in x
        kp2[10] = [0.8, 0.5, 1.0]  # -0.1 in x

        lm = Landmarks(kp1)
        lm.update(kp2)
        mean_vel = lm.mean_velocity([9, 10])
        assert mean_vel == pytest.approx(0.1, abs=1e-5)

    def test_height_is_inverted_y(self):
        """In image coords, y=0 is top. Height should be 1-y so arms up = high."""
        kp = self._make_keypoints()
        lm = Landmarks(kp)
        # Wrists at y=0.5, so height = 1 - 0.5 = 0.5
        assert lm.height(9) == pytest.approx(0.5)
        # Ears at y=0.1, so height = 0.9
        assert lm.height(3) == pytest.approx(0.9)
