import sys
import numpy as np
import pytest
from unittest.mock import MagicMock

# Mock ultralytics before importing detector
mock_ultralytics = MagicMock()
sys.modules["ultralytics"] = mock_ultralytics

from vision.detector import PoseDetector  # noqa: E402


class TestPoseDetector:
    def _make_mock_result(self, num_people=1):
        """Create a mock ultralytics result with keypoints."""
        result = MagicMock()
        if num_people == 0:
            result.keypoints = None
            return result

        kp_data = np.random.rand(num_people, 17, 3).astype(np.float32)
        kp_data[:, :, 2] = 0.9  # high confidence
        result.keypoints = MagicMock()
        result.keypoints.data = MagicMock()
        result.keypoints.data.cpu.return_value.numpy.return_value = kp_data
        return result

    def test_detect_returns_list_of_keypoints(self):
        mock_model = MagicMock()
        mock_ultralytics.YOLO.return_value = mock_model
        mock_model.return_value = [self._make_mock_result(num_people=1)]

        detector = PoseDetector(model_path="yolov8n-pose.pt", confidence=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(frame)

        assert len(results) == 1
        assert results[0].shape == (17, 3)

    def test_detect_multiple_people(self):
        mock_model = MagicMock()
        mock_ultralytics.YOLO.return_value = mock_model
        mock_model.return_value = [self._make_mock_result(num_people=3)]

        detector = PoseDetector(model_path="yolov8n-pose.pt", confidence=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(frame)

        assert len(results) == 3

    def test_detect_no_people(self):
        mock_model = MagicMock()
        mock_ultralytics.YOLO.return_value = mock_model
        mock_model.return_value = [self._make_mock_result(num_people=0)]

        detector = PoseDetector(model_path="yolov8n-pose.pt", confidence=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(frame)

        assert len(results) == 0

    def test_normalises_coordinates(self):
        mock_model = MagicMock()
        mock_ultralytics.YOLO.return_value = mock_model

        result = MagicMock()
        kp_data = np.zeros((1, 17, 3), dtype=np.float32)
        kp_data[0, 0] = [320, 240, 0.9]  # nose at center of 640x480
        result.keypoints = MagicMock()
        result.keypoints.data = MagicMock()
        result.keypoints.data.cpu.return_value.numpy.return_value = kp_data
        mock_model.return_value = [result]

        detector = PoseDetector(model_path="yolov8n-pose.pt", confidence=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(frame)

        # x=320/640=0.5, y=240/480=0.5
        assert results[0][0, 0] == pytest.approx(0.5, abs=0.01)
        assert results[0][0, 1] == pytest.approx(0.5, abs=0.01)
