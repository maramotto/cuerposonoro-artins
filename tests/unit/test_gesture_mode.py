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


def _make_landmarks_stream(dy: float, n_frames: int) -> list[Landmarks]:
    """Create a stream of landmarks with consistent vertical movement.

    Returns a single Landmarks object updated across n_frames, simulating
    consecutive frames from the same person. Each frame moves keypoints
    by dy in the y-axis.
    """
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, 2] = 1.0
    kp[10] = [0.5, 0.5, 1.0]  # right wrist
    kp[8] = [0.5, 0.4, 1.0]   # right elbow
    kp[9] = [0.3, 0.5, 1.0]   # left wrist
    kp[7] = [0.3, 0.4, 1.0]   # left elbow
    kp[11] = [0.4, 0.6, 1.0]  # left hip
    kp[12] = [0.6, 0.6, 1.0]  # right hip

    lm = Landmarks(kp.copy())
    frames = [lm]

    for i in range(1, n_frames):
        next_kp = kp.copy()
        for idx in [7, 8, 9, 10, 11, 12]:
            next_kp[idx, 1] += dy * i
        lm.update(next_kp)
        frames.append(lm)

    return frames


class TestGestureModeCooldown:
    def test_first_note_triggers_immediately(self):
        """First note should trigger without any cooldown delay."""
        synth = MagicMock()
        mode = GestureMidiMode(synth=synth, config=_config(cooldown_ms=500))

        frames = _make_landmarks_stream(dy=-0.05, n_frames=4)
        for lm in frames:
            mode.update(lm)

        assert synth.noteon.called

    def test_rapid_updates_are_rate_limited(self):
        """Notes should not retrigger faster than cooldown_ms."""
        synth = MagicMock()
        mode = GestureMidiMode(synth=synth, config=_config(cooldown_ms=500))

        frames = _make_landmarks_stream(dy=-0.05, n_frames=4)
        for lm in frames:
            mode.update(lm)

        first_count = synth.noteon.call_count

        # Reverse direction — still within cooldown
        frames2 = _make_landmarks_stream(dy=0.05, n_frames=4)
        for lm in frames2:
            mode.update(lm)

        assert synth.noteon.call_count == first_count

    def test_note_triggers_after_cooldown_expires(self):
        """After cooldown period, new notes should trigger."""
        synth = MagicMock()
        mode = GestureMidiMode(synth=synth, config=_config(cooldown_ms=50))

        frames = _make_landmarks_stream(dy=-0.05, n_frames=4)
        for lm in frames:
            mode.update(lm)

        first_count = synth.noteon.call_count
        assert first_count > 0

        time.sleep(0.06)

        frames2 = _make_landmarks_stream(dy=0.05, n_frames=4)
        for lm in frames2:
            mode.update(lm)

        assert synth.noteon.call_count > first_count

    def test_zero_cooldown_allows_every_note(self):
        """With cooldown_ms=0, no rate limiting should occur."""
        synth = MagicMock()
        mode = GestureMidiMode(synth=synth, config=_config(cooldown_ms=0))

        frames = _make_landmarks_stream(dy=-0.05, n_frames=6)
        for lm in frames:
            mode.update(lm)

        assert synth.noteon.call_count >= 2
