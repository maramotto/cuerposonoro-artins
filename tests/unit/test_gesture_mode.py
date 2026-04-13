"""Tests for GestureMidiMode — focus on note cooldown (rate limiting)."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from audio.gesture_mode import GestureMidiMode
from vision.landmarks import Landmarks


def _config(cooldown_ms: int = 150) -> dict:
    return {
        "gesture": {
            "min_velocity": 0.005,
            "hysteresis_frames": 2,
            "silence_frames": 18,
            "velocity_max": 0.15,
            "staccato_frames": 3,
            "sustain_frames": 4,
            "reverb_cc91": 80,
            "note_cooldown_ms": cooldown_ms,
            "programs": {
                "arms": {"channel": 0, "program": 11},
                "center": {"channel": 1, "program": 32},
            },
        },
    }


def _moving_landmarks(dy: float = -0.05) -> Landmarks:
    """Create landmarks with vertical movement (dy < 0 = upward)."""
    kp1 = np.zeros((17, 3), dtype=np.float32)
    kp1[:, 2] = 1.0
    kp1[10] = [0.5, 0.5, 1.0]  # right wrist
    kp1[8] = [0.5, 0.4, 1.0]   # right elbow
    kp1[9] = [0.3, 0.5, 1.0]   # left wrist
    kp1[7] = [0.3, 0.4, 1.0]   # left elbow
    kp1[11] = [0.4, 0.6, 1.0]  # left hip
    kp1[12] = [0.6, 0.6, 1.0]  # right hip

    kp2 = kp1.copy()
    for i in [7, 8, 9, 10, 11, 12]:
        kp2[i, 1] += dy

    lm = Landmarks(kp1)
    lm.update(kp2)
    return lm


class TestGestureModeCooldown:
    def test_first_note_triggers_immediately(self):
        """First note should trigger without any cooldown delay."""
        synth = MagicMock()
        mode = GestureMidiMode(synth=synth, config=_config(cooldown_ms=500))

        for _ in range(3):
            lm = _moving_landmarks(dy=-0.05)
            mode.update(lm)

        assert synth.noteon.called

    def test_rapid_updates_are_rate_limited(self):
        """Notes should not retrigger faster than cooldown_ms."""
        synth = MagicMock()
        mode = GestureMidiMode(synth=synth, config=_config(cooldown_ms=500))

        for _ in range(3):
            mode.update(_moving_landmarks(dy=-0.05))

        first_count = synth.noteon.call_count

        for _ in range(3):
            mode.update(_moving_landmarks(dy=0.05))

        assert synth.noteon.call_count == first_count

    def test_note_triggers_after_cooldown_expires(self):
        """After cooldown period, new notes should trigger."""
        synth = MagicMock()
        mode = GestureMidiMode(synth=synth, config=_config(cooldown_ms=50))

        for _ in range(3):
            mode.update(_moving_landmarks(dy=-0.05))

        first_count = synth.noteon.call_count
        assert first_count > 0

        time.sleep(0.06)

        for _ in range(3):
            mode.update(_moving_landmarks(dy=0.05))

        assert synth.noteon.call_count > first_count

    def test_zero_cooldown_allows_every_note(self):
        """With cooldown_ms=0, no rate limiting should occur."""
        synth = MagicMock()
        mode = GestureMidiMode(synth=synth, config=_config(cooldown_ms=0))

        for _ in range(6):
            mode.update(_moving_landmarks(dy=-0.05))

        assert synth.noteon.call_count >= 2
