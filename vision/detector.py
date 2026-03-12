from __future__ import annotations

import numpy as np
from ultralytics import YOLO


class PoseDetector:
    """YOLOv8-Pose wrapper that returns normalised keypoints."""

    def __init__(self, model_path: str, confidence: float) -> None:
        self._model = YOLO(model_path)
        self._confidence = confidence

    def detect(self, frame: np.ndarray) -> list[np.ndarray]:
        """Run pose detection on a frame.

        Returns a list of (17, 3) arrays — one per detected person.
        Coordinates are normalised to [0, 1] relative to frame dimensions.
        """
        results = self._model(frame, conf=self._confidence, verbose=False)
        result = results[0]

        if result.keypoints is None:
            return []

        keypoints = result.keypoints.data.cpu().numpy()
        if keypoints.ndim != 3 or keypoints.shape[0] == 0:
            return []

        h, w = frame.shape[:2]
        people = []
        for person_kp in keypoints:
            normalised = person_kp.copy()
            normalised[:, 0] /= w
            normalised[:, 1] /= h
            people.append(normalised)

        return people
