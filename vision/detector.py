from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from ultralytics import YOLO

log = logging.getLogger(__name__)


class PoseDetector:
    """YOLOv8-Pose wrapper that returns normalised keypoints.

    Supports TensorRT acceleration: when ``use_tensorrt=True`` (the default),
    the detector looks for a pre-built ``.engine`` file next to the ``.pt``
    model.  If the engine exists it is loaded directly; otherwise the ``.pt``
    model is exported to TensorRT first (takes ~2 min on the Jetson).  If the
    export fails the detector falls back to the ``.pt`` model on CPU.
    """

    def __init__(
        self,
        model_path: str,
        confidence: float,
        use_tensorrt: bool = True,
        tensorrt_half: bool = True,
    ) -> None:
        self._confidence = confidence
        self._model = self._load_model(
            model_path, use_tensorrt, tensorrt_half,
        )

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

    @staticmethod
    def _load_model(
        model_path: str,
        use_tensorrt: bool,
        tensorrt_half: bool,
    ) -> YOLO:
        if not use_tensorrt:
            log.info("Loading model from %s (TensorRT disabled)", model_path)
            return YOLO(model_path)

        engine_path = Path(model_path).with_suffix(".engine")

        if engine_path.exists():
            log.info("Loading TensorRT engine from %s", engine_path)
            return YOLO(str(engine_path))

        # Export .pt → .engine
        log.info(
            "Exporting to TensorRT (first run, this takes ~2min): %s → %s",
            model_path,
            engine_path,
        )
        try:
            pt_model = YOLO(model_path)
            pt_model.export(format="engine", half=tensorrt_half, device=0)
            log.info("TensorRT export complete, loading engine")
            return YOLO(str(engine_path))
        except (RuntimeError, OSError):
            log.exception(
                "TensorRT export failed, falling back to CPU with %s",
                model_path,
            )
            return YOLO(model_path)
