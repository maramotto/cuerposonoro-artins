import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call


from vision.capture import WebcamCamera


class TestWebcamCamera:
    @patch("vision.capture.cv2")
    def test_read_returns_ndarray_on_success(self, mock_cv2):
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)
        mock_cap.grab.return_value = False  # no buffered frames

        camera = WebcamCamera(device_id=0)
        result = camera.read()

        assert result is not None
        assert isinstance(result, np.ndarray)

    @patch("vision.capture.cv2")
    def test_read_returns_none_on_failure(self, mock_cv2):
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)

        camera = WebcamCamera(device_id=0)
        result = camera.read()

        assert result is None

    @patch("vision.capture.cv2")
    def test_read_drains_buffer(self, mock_cv2):
        """read() calls grab() after initial read to drain stale frames."""
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True

        first_frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        mock_cap.read.return_value = (True, first_frame)
        # Two buffered frames, then buffer empty
        mock_cap.grab.side_effect = [True, True, False]
        newer_frame_1 = np.ones((480, 640, 3), dtype=np.uint8) * 200
        newer_frame_2 = np.ones((480, 640, 3), dtype=np.uint8) * 255
        mock_cap.retrieve.side_effect = [
            (True, newer_frame_1),
            (True, newer_frame_2),
        ]

        camera = WebcamCamera(device_id=0)
        result = camera.read()

        # Must return the newest frame, not the first one
        assert np.array_equal(result, newer_frame_2)
        assert mock_cap.grab.call_count == 3

    @patch("vision.capture.cv2")
    def test_read_returns_newest_frame(self, mock_cv2):
        """When buffer has stale frames, read() returns only the latest."""
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True

        stale = np.zeros((480, 640, 3), dtype=np.uint8)
        newest = np.ones((480, 640, 3), dtype=np.uint8) * 42
        mock_cap.read.return_value = (True, stale)
        mock_cap.grab.side_effect = [True, False]
        mock_cap.retrieve.return_value = (True, newest)

        camera = WebcamCamera(device_id=0)
        result = camera.read()

        assert np.array_equal(result, newest)

    @patch("vision.capture.cv2")
    def test_read_keeps_first_frame_if_retrieve_fails(self, mock_cv2):
        """If grab succeeds but retrieve fails, keep the previous good frame."""
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True

        good_frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
        mock_cap.read.return_value = (True, good_frame)
        mock_cap.grab.side_effect = [True, False]
        mock_cap.retrieve.return_value = (False, None)

        camera = WebcamCamera(device_id=0)
        result = camera.read()

        assert np.array_equal(result, good_frame)

    @patch("vision.capture.cv2")
    def test_release_calls_cap_release(self, mock_cv2):
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True

        camera = WebcamCamera(device_id=0)
        camera.release()

        mock_cap.release.assert_called_once()

    @patch("vision.capture.cv2")
    def test_constructor_opens_camera(self, mock_cv2):
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True

        camera = WebcamCamera(device_id=2)

        mock_cv2.VideoCapture.assert_called_once_with(2)
