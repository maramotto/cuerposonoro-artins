from __future__ import annotations
import logging
import cv2
import numpy as np
log = logging.getLogger(__name__)

class WebcamCamera:
    """Wraps cv2.VideoCapture for real-time pipelines."""
    def __init__(self, device_id: int) -> None:
        self._cap = cv2.VideoCapture(device_id)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> np.ndarray | None:
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def release(self) -> None:
        self._cap.release()

    @property
    def is_opened(self) -> bool:
        return self._cap.isOpened()
