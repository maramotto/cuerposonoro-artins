from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

log = logging.getLogger(__name__)


class WebcamCamera:
    """Wraps cv2.VideoCapture with synchronous buffer flushing."""

    def __init__(self, device_id: int) -> None:
        self._cap = cv2.VideoCapture(device_id)

    def read(self) -> np.ndarray | None:
        """Return the most recent frame, flushing any stale buffered frames."""
        ret, frame = self._cap.read()
        if not ret:
            return None

        while self._cap.grab():
            ret, newer = self._cap.retrieve()
            if ret:
                frame = newer

        return frame

    def release(self) -> None:
        self._cap.release()

    @property
    def is_opened(self) -> bool:
        return self._cap.isOpened()


class ThreadedCamera:
    """Background-threaded camera that always provides the most recent frame.

    A dedicated thread continuously reads from the camera, storing only the
    latest frame in a single slot. The main thread's read() returns immediately
    with whatever frame is newest, never blocking on the camera.
    """

    def __init__(self, device_id: int) -> None:
        self._cap = cv2.VideoCapture(device_id)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def read(self) -> np.ndarray | None:
        """Return the most recent frame without blocking, or None."""
        with self._lock:
            return self._frame

    def release(self) -> None:
        """Stop the capture thread and release the camera."""
        self._running = False
        self._thread.join(timeout=2.0)
        self._cap.release()

    @property
    def is_opened(self) -> bool:
        return self._cap.isOpened()
