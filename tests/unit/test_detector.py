import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

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


class TestTensorRTLoading:
    """Tests for TensorRT engine export and loading logic."""

    def setup_method(self):
        mock_ultralytics.YOLO.reset_mock()
        mock_ultralytics.YOLO.side_effect = None

    @patch("vision.detector.Path")
    def test_loads_engine_when_exists(self, mock_path_cls):
        """When use_tensorrt=True and .engine exists, load it directly."""
        mock_engine_path = MagicMock()
        mock_engine_path.exists.return_value = True
        mock_engine_path.__str__ = lambda self: "yolov8n-pose.engine"
        mock_path_cls.return_value.with_suffix.return_value = mock_engine_path

        mock_model = MagicMock()
        mock_ultralytics.YOLO.return_value = mock_model

        detector = PoseDetector(
            model_path="yolov8n-pose.pt",
            confidence=0.5,
            use_tensorrt=True,
        )

        # YOLO should be called with the .engine path, not .pt
        mock_ultralytics.YOLO.assert_called_with("yolov8n-pose.engine")
        # export should NOT be called
        mock_model.export.assert_not_called()

    @patch("vision.detector.Path")
    def test_exports_engine_when_not_exists(self, mock_path_cls):
        """When use_tensorrt=True and .engine missing, export then load."""
        mock_engine_path = MagicMock()
        mock_engine_path.exists.return_value = False
        mock_engine_path.__str__ = lambda self: "yolov8n-pose.engine"
        mock_path_cls.return_value.with_suffix.return_value = mock_engine_path

        mock_pt_model = MagicMock()
        mock_engine_model = MagicMock()
        # First YOLO() loads .pt, second loads the exported .engine
        mock_ultralytics.YOLO.side_effect = [mock_pt_model, mock_engine_model]
        mock_pt_model.export.return_value = "yolov8n-pose.engine"

        detector = PoseDetector(
            model_path="yolov8n-pose.pt",
            confidence=0.5,
            use_tensorrt=True,
            tensorrt_half=True,
        )

        # Should load .pt first, export, then load .engine
        mock_pt_model.export.assert_called_once_with(
            format="engine", half=True, device=0,
        )
        assert mock_ultralytics.YOLO.call_count == 2

    @patch("vision.detector.Path")
    def test_no_tensorrt_loads_pt_directly(self, mock_path_cls):
        """When use_tensorrt=False, load .pt directly without export."""
        mock_model = MagicMock()
        mock_ultralytics.YOLO.return_value = mock_model

        detector = PoseDetector(
            model_path="yolov8n-pose.pt",
            confidence=0.5,
            use_tensorrt=False,
        )

        mock_ultralytics.YOLO.assert_called_once_with("yolov8n-pose.pt")
        mock_model.export.assert_not_called()

    @patch("vision.detector.Path")
    def test_default_use_tensorrt_is_true(self, mock_path_cls):
        """use_tensorrt defaults to True."""
        mock_engine_path = MagicMock()
        mock_engine_path.exists.return_value = True
        mock_engine_path.__str__ = lambda self: "yolov8n-pose.engine"
        mock_path_cls.return_value.with_suffix.return_value = mock_engine_path

        mock_ultralytics.YOLO.return_value = MagicMock()

        detector = PoseDetector(
            model_path="yolov8n-pose.pt",
            confidence=0.5,
        )

        # Should load the engine (TensorRT is default)
        mock_ultralytics.YOLO.assert_called_with("yolov8n-pose.engine")

    @patch("vision.detector.Path")
    def test_export_failure_falls_back_to_cpu(self, mock_path_cls):
        """If TensorRT export fails, fall back to .pt model without crashing."""
        mock_engine_path = MagicMock()
        mock_engine_path.exists.return_value = False
        mock_engine_path.__str__ = lambda self: "yolov8n-pose.engine"
        mock_path_cls.return_value.with_suffix.return_value = mock_engine_path

        mock_pt_model = MagicMock()
        mock_pt_model.export.side_effect = RuntimeError("TensorRT not available")
        mock_ultralytics.YOLO.return_value = mock_pt_model

        # Should NOT raise — falls back gracefully
        detector = PoseDetector(
            model_path="yolov8n-pose.pt",
            confidence=0.5,
            use_tensorrt=True,
        )

        # Model should still be the .pt model (fallback)
        mock_ultralytics.YOLO.assert_called_with("yolov8n-pose.pt")
