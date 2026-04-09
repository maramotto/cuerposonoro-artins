import time

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


from vision.capture import WebcamCamera, ThreadedCamera


# --- WebcamCamera tests ---

class TestWebcamCamera:
    @patch("vision.capture.cv2")
    def test_read_returns_ndarray_on_success(self, mock_cv2):
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)
        mock_cap.grab.return_value = False

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
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True

        first_frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        mock_cap.read.return_value = (True, first_frame)
        mock_cap.grab.side_effect = [True, True, False]
        newer_frame_1 = np.ones((480, 640, 3), dtype=np.uint8) * 200
        newer_frame_2 = np.ones((480, 640, 3), dtype=np.uint8) * 255
        mock_cap.retrieve.side_effect = [
            (True, newer_frame_1),
            (True, newer_frame_2),
        ]

        camera = WebcamCamera(device_id=0)
        result = camera.read()

        assert np.array_equal(result, newer_frame_2)
        assert mock_cap.grab.call_count == 3

    @patch("vision.capture.cv2")
    def test_release_calls_cap_release(self, mock_cv2):
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True

        camera = WebcamCamera(device_id=0)
        camera.release()

        mock_cap.release.assert_called_once()


# --- ThreadedCamera tests ---

class TestThreadedCamera:
    @patch("vision.capture.cv2")
    def test_read_returns_none_before_any_frame(self, mock_cv2):
        """read() returns None if the background thread hasn't captured yet."""
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        # Block cap.read so the thread never produces a frame
        mock_cap.read.return_value = (False, None)

        camera = ThreadedCamera(device_id=0)
        result = camera.read()

        assert result is None
        camera.release()

    @patch("vision.capture.cv2")
    def test_read_returns_most_recent_frame(self, mock_cv2):
        """read() returns the latest frame captured by the background thread."""
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 42
        mock_cap.read.return_value = (True, frame)

        camera = ThreadedCamera(device_id=0)
        # Give the background thread time to capture at least one frame
        time.sleep(0.1)
        result = camera.read()

        assert result is not None
        assert np.array_equal(result, frame)
        camera.release()

    @patch("vision.capture.cv2")
    def test_release_stops_background_thread(self, mock_cv2):
        """release() stops the capture thread and releases the camera."""
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))

        camera = ThreadedCamera(device_id=0)
        camera.release()

        mock_cap.release.assert_called_once()
        # Thread should no longer be alive
        assert not camera._thread.is_alive()

    @patch("vision.capture.cv2")
    def test_old_frames_are_discarded(self, mock_cv2):
        """Only the latest frame is kept; older frames are overwritten."""
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True

        # Simulate a sequence of frames
        frame_old = np.ones((480, 640, 3), dtype=np.uint8) * 10
        frame_new = np.ones((480, 640, 3), dtype=np.uint8) * 200
        call_count = 0

        def fake_read():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return (True, frame_old.copy())
            return (True, frame_new.copy())

        mock_cap.read.side_effect = fake_read

        camera = ThreadedCamera(device_id=0)
        # Wait for several frames to be captured
        time.sleep(0.15)
        result = camera.read()

        assert result is not None
        # The returned frame should be the newest one, not an old one
        assert np.array_equal(result, frame_new)
        camera.release()

    @patch("vision.capture.cv2")
    def test_is_opened_property(self, mock_cv2):
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)

        camera = ThreadedCamera(device_id=0)
        assert camera.is_opened

        mock_cap.isOpened.return_value = False
        assert not camera.is_opened
        camera.release()

    @patch("vision.capture.cv2")
    def test_constructor_opens_camera(self, mock_cv2):
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)

        camera = ThreadedCamera(device_id=2)

        mock_cv2.VideoCapture.assert_called_once_with(2)
        camera.release()
