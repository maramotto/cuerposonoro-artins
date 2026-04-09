import numpy as np
from unittest.mock import MagicMock, patch

from vision.capture import WebcamCamera


class TestWebcamCamera:
    @patch("vision.capture.cv2")
    def test_read_returns_ndarray_on_success(self, mock_cv2):
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)

        camera = WebcamCamera(device_id=0)
        result = camera.read()

        assert result is not None
        assert isinstance(result, np.ndarray)

    @patch("vision.capture.cv2")
    def test_read_returns_none_on_failure(self, mock_cv2):
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.read.return_value = (False, None)

        camera = WebcamCamera(device_id=0)
        result = camera.read()

        assert result is None

    @patch("vision.capture.cv2")
    def test_sets_buffer_size_to_one(self, mock_cv2):
        """Constructor sets CAP_PROP_BUFFERSIZE=1 to minimise latency."""
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap

        WebcamCamera(device_id=0)

        mock_cap.set.assert_called_once_with(mock_cv2.CAP_PROP_BUFFERSIZE, 1)

    @patch("vision.capture.cv2")
    def test_release_calls_cap_release(self, mock_cv2):
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap

        camera = WebcamCamera(device_id=0)
        camera.release()

        mock_cap.release.assert_called_once()

    @patch("vision.capture.cv2")
    def test_is_opened_property(self, mock_cv2):
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True

        camera = WebcamCamera(device_id=0)
        assert camera.is_opened

        mock_cap.isOpened.return_value = False
        assert not camera.is_opened

    @patch("vision.capture.cv2")
    def test_constructor_opens_camera(self, mock_cv2):
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap

        WebcamCamera(device_id=2)

        mock_cv2.VideoCapture.assert_called_once_with(2)
