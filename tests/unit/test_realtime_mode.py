"""Tests for RealtimeMidiMode — focus on note cooldown (rate limiting)."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from audio.realtime_mode import RealtimeMidiMode
from vision.landmarks import Landmarks


def _config(cooldown_ms: int = 150) -> dict:
    return {
        "realtime": {
            "min_velocity": 0.007,
            "hysteresis_frames": 2,
            "silence_frames": 18,
            "velocity_max": 0.15,
            "midi_velocity_min": 40,
            "midi_velocity_max": 120,
            "note_cooldown_ms": cooldown_ms,
            "programs": {
                "melody": {"channel": 0, "program": 11},
                "bass": {"channel": 1, "program": 32},
            },
            "keypoints": {
                "right_wrist": {
                    "index": 10,
                    "channel": 0,
                    "note_min": 74,
                    "note_max": 86,
                },
            },
        },
    }


def _moving_landmarks(dx: float = 0.1) -> Landmarks:
    kp1 = np.zeros((17, 3), dtype=np.float32)
    kp1[:, 2] = 1.0
    kp1[10] = [0.5, 0.5, 1.0]

    kp2 = kp1.copy()
    kp2[10, 0] += dx

    lm = Landmarks(kp1)
    lm.update(kp2)
    return lm


class TestRealtimeModeCooldown:
    def test_first_note_triggers(self):
        synth = MagicMock()
        mode = RealtimeMidiMode(synth=synth, config=_config(cooldown_ms=0))

        for _ in range(3):
            mode.update(_moving_landmarks(dx=0.1))

        assert synth.noteon.called

    def test_rapid_updates_rate_limited(self):
        synth = MagicMock()
        mode = RealtimeMidiMode(synth=synth, config=_config(cooldown_ms=500))

        for _ in range(3):
            mode.update(_moving_landmarks(dx=0.1))

        first_count = synth.noteon.call_count

        for _ in range(3):
            mode.update(_moving_landmarks(dx=-0.1))

        assert synth.noteon.call_count == first_count

    def test_triggers_after_cooldown(self):
        synth = MagicMock()
        mode = RealtimeMidiMode(synth=synth, config=_config(cooldown_ms=50))

        # Small movement → low note in pentatonic range
        for _ in range(3):
            mode.update(_moving_landmarks(dx=0.02))

        first_count = synth.noteon.call_count

        time.sleep(0.06)

        # Large movement after cooldown → different (higher) note
        for _ in range(3):
            mode.update(_moving_landmarks(dx=0.14))

        assert synth.noteon.call_count > first_count
