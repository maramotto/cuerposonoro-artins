from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


class WebcamCamera:
    """Wraps cv2.VideoCapture with buffer flushing for real-time pipelines."""

    def __init__(self, device_id: int) -> None:
        self._cap = cv2.VideoCapture(device_id)

    def read(self) -> np.ndarray | None:
        """Return the most recent frame, flushing any stale buffered frames."""
        ret, frame = self._cap.read()
        if not ret:
            return None

        # Drain buffered frames — grab() is cheap (no decode),
        # retrieve() only decodes the last grabbed frame.
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
