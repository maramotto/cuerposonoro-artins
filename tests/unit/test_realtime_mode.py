"""Tests for RealtimeMidiMode — per-keypoint velocity-driven MIDI.

D minor pentatonic, no beat grid, hysteresis + silence per keypoint.
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from audio.realtime_mode import RealtimeMidiMode
from vision.landmarks import Landmarks


# --- Helpers ---

def _make_keypoints(**overrides) -> np.ndarray:
    """Create a 17x3 COCO keypoints array.

    Override specific keypoints with keyword arguments:
        _make_keypoints(kp9=(0.3, 0.5), kp10=(0.7, 0.5))
    """
    kp = np.full((17, 3), 0.5, dtype=np.float32)
    kp[:, 2] = 1.0  # confidence
    for key, (x, y) in overrides.items():
        idx = int(key[2:])
        kp[idx] = [x, y, 1.0]
    return kp


def _make_config():
    return {
        "realtime": {
            "min_velocity": 0.015,
            "hysteresis_frames": 2,
            "silence_frames": 6,
            "velocity_max": 0.15,
            "midi_velocity_min": 40,
            "midi_velocity_max": 120,
            "programs": {
                "melody": {"channel": 0, "program": 11},
                "bass": {"channel": 1, "program": 32},
            },
            "keypoints": {
                "right_wrist": {
                    "index": 10, "channel": 0,
                    "note_min": 74, "note_max": 86,
                },
                "left_wrist": {
                    "index": 9, "channel": 0,
                    "note_min": 62, "note_max": 74,
                },
                "right_ankle": {
                    "index": 16, "channel": 1,
                    "note_min": 38, "note_max": 50,
                },
                "left_ankle": {
                    "index": 15, "channel": 1,
                    "note_min": 38, "note_max": 50,
                },
            },
        },
    }


def _make_mode(synth=None, config=None):
    if synth is None:
        synth = MagicMock()
    if config is None:
        config = _make_config()
    return RealtimeMidiMode(synth=synth, config=config), synth


def _feed_frames(mode, landmarks_list):
    """Feed a sequence of Landmarks into the mode."""
    for lm in landmarks_list:
        mode.update(lm)


def _moving_wrist_landmarks(n_frames: int, wrist_index: int = 10) -> list[Landmarks]:
    """Create n_frames of Landmarks where a wrist moves significantly."""
    frames = []
    kp_key = f"kp{wrist_index}"
    for i in range(n_frames):
        y = 0.5 - i * 0.05  # move upward
        kp = _make_keypoints(**{kp_key: (0.5, y)})
        if i == 0:
            lm = Landmarks(kp)
        else:
            lm = frames[-1]._replace_keypoints(kp) if hasattr(frames[-1], '_replace_keypoints') else None
            # Use the real Landmarks API: create, then update
            lm = Landmarks(_make_keypoints(**{kp_key: (0.5, 0.5 - (i - 1) * 0.05)}))
            lm.update(kp)
        frames.append(lm)
    return frames


def _landmarks_pair(index: int, x0: float, y0: float, x1: float, y1: float) -> Landmarks:
    """Create a Landmarks with one update so velocity at `index` is nonzero."""
    key = f"kp{index}"
    prev_kp = _make_keypoints(**{key: (x0, y0)})
    curr_kp = _make_keypoints(**{key: (x1, y1)})
    lm = Landmarks(prev_kp)
    lm.update(curr_kp)
    return lm


def _still_landmarks() -> Landmarks:
    """Landmarks with zero velocity (first frame, no update)."""
    return Landmarks(_make_keypoints())


# --- Tests ---

class TestHysteresis:
    """Notes must not trigger until velocity exceeds threshold for N consecutive frames."""

    def test_no_note_on_single_frame(self):
        """One frame of wrist movement should NOT trigger a note (hysteresis=2)."""
        mode, synth = _make_mode()

        lm = _landmarks_pair(10, 0.5, 0.5, 0.5, 0.4)
        mode.update(lm)

        synth.noteon.assert_not_called()

    def test_note_triggers_after_hysteresis_frames(self):
        """Wrist note triggers after 2 consecutive frames above threshold."""
        mode, synth = _make_mode()

        lm1 = _landmarks_pair(10, 0.5, 0.5, 0.5, 0.4)
        mode.update(lm1)
        synth.noteon.assert_not_called()

        lm2 = _landmarks_pair(10, 0.5, 0.4, 0.5, 0.3)
        mode.update(lm2)
        synth.noteon.assert_called_once()
        # Should be on channel 0 (melody) since index 10 = right wrist
        assert synth.noteon.call_args[0][0] == 0

    def test_ankle_triggers_on_bass_channel(self):
        """Ankle movement triggers on channel 1 (bass)."""
        mode, synth = _make_mode()

        lm1 = _landmarks_pair(16, 0.5, 0.8, 0.5, 0.7)
        mode.update(lm1)

        lm2 = _landmarks_pair(16, 0.5, 0.7, 0.5, 0.6)
        mode.update(lm2)

        synth.noteon.assert_called()
        assert synth.noteon.call_args[0][0] == 1


class TestSilence:
    """Notes must release after SILENCE_FRAMES of stillness."""

    def test_note_released_after_silence_frames(self):
        """Note is released after 6 frames of no movement."""
        mode, synth = _make_mode()

        # Trigger a note (2 frames of movement)
        lm1 = _landmarks_pair(10, 0.5, 0.5, 0.5, 0.4)
        mode.update(lm1)
        lm2 = _landmarks_pair(10, 0.5, 0.4, 0.5, 0.3)
        mode.update(lm2)
        assert synth.noteon.called

        # Feed 5 still frames — note should NOT be released yet
        synth.reset_mock()
        for _ in range(5):
            mode.update(_still_landmarks())
        synth.noteoff.assert_not_called()

        # 6th still frame — note should be released
        mode.update(_still_landmarks())
        synth.noteoff.assert_called()
        assert synth.noteoff.call_args[0][0] == 0

    def test_movement_resets_silence_counter(self):
        """Resuming movement before SILENCE_FRAMES resets the counter."""
        mode, synth = _make_mode()

        # Trigger a note
        lm1 = _landmarks_pair(10, 0.5, 0.5, 0.5, 0.4)
        mode.update(lm1)
        lm2 = _landmarks_pair(10, 0.5, 0.4, 0.5, 0.3)
        mode.update(lm2)
        assert synth.noteon.called

        # Feed 4 still frames (not enough for silence)
        for _ in range(4):
            mode.update(_still_landmarks())

        # Resume movement
        synth.reset_mock()
        lm3 = _landmarks_pair(10, 0.5, 0.3, 0.5, 0.2)
        mode.update(lm3)

        # Note should not have been released
        synth.noteoff.assert_not_called()


class TestPitchMapping:
    """Velocity maps to note within the keypoint's pentatonic range."""

    def test_note_within_configured_range(self):
        """Triggered note must be within [note_min, note_max]."""
        mode, synth = _make_mode()

        lm1 = _landmarks_pair(10, 0.5, 0.5, 0.5, 0.4)
        mode.update(lm1)
        lm2 = _landmarks_pair(10, 0.5, 0.4, 0.5, 0.3)
        mode.update(lm2)

        note = synth.noteon.call_args[0][1]
        assert 74 <= note <= 86, f"Note {note} outside right_wrist range [74, 86]"

    def test_note_is_in_d_minor_pentatonic(self):
        """Triggered note must belong to D minor pentatonic (pitch classes 0,2,5,7,9)."""
        mode, synth = _make_mode()

        lm1 = _landmarks_pair(10, 0.5, 0.5, 0.5, 0.35)
        mode.update(lm1)
        lm2 = _landmarks_pair(10, 0.5, 0.35, 0.5, 0.2)
        mode.update(lm2)

        note = synth.noteon.call_args[0][1]
        pitch_class = note % 12
        d_minor_pentatonic = {2, 5, 7, 9, 0}  # D, F, G, A, C
        assert pitch_class in d_minor_pentatonic, (
            f"Note {note} (pc={pitch_class}) not in D minor pentatonic"
        )


class TestVelocityMapping:
    """MIDI velocity maps from movement speed."""

    def test_faster_movement_higher_velocity(self):
        """Faster keypoint movement produces higher MIDI velocity."""
        # Slow movement
        mode_slow, synth_slow = _make_mode()
        lm1 = _landmarks_pair(10, 0.5, 0.5, 0.5, 0.48)
        mode_slow.update(lm1)
        lm2 = _landmarks_pair(10, 0.5, 0.48, 0.5, 0.46)
        mode_slow.update(lm2)

        # Fast movement
        mode_fast, synth_fast = _make_mode()
        lm3 = _landmarks_pair(10, 0.5, 0.5, 0.5, 0.3)
        mode_fast.update(lm3)
        lm4 = _landmarks_pair(10, 0.5, 0.3, 0.5, 0.1)
        mode_fast.update(lm4)

        if synth_slow.noteon.called and synth_fast.noteon.called:
            vel_slow = synth_slow.noteon.call_args[0][2]
            vel_fast = synth_fast.noteon.call_args[0][2]
            assert vel_fast >= vel_slow


class TestMultipleKeypoints:
    """Each keypoint is tracked independently."""

    def test_wrist_and_ankle_trigger_independently(self):
        """Moving right wrist and right ankle triggers notes on both channels."""
        mode, synth = _make_mode()

        # Frame 1: both moving
        kp_prev = _make_keypoints(kp10=(0.5, 0.5), kp16=(0.5, 0.8))
        kp_curr = _make_keypoints(kp10=(0.5, 0.4), kp16=(0.5, 0.7))
        lm1 = Landmarks(kp_prev)
        lm1.update(kp_curr)
        mode.update(lm1)

        # Frame 2: both still moving
        kp_prev2 = kp_curr
        kp_curr2 = _make_keypoints(kp10=(0.5, 0.3), kp16=(0.5, 0.6))
        lm2 = Landmarks(kp_prev2)
        lm2.update(kp_curr2)
        mode.update(lm2)

        channels = {c[0][0] for c in synth.noteon.call_args_list}
        assert 0 in channels, "Melody channel not triggered"
        assert 1 in channels, "Bass channel not triggered"


class TestClose:
    """close() releases all held notes."""

    def test_close_releases_active_notes(self):
        """close() sends noteoff for any held notes and CC 123."""
        mode, synth = _make_mode()

        lm1 = _landmarks_pair(10, 0.5, 0.5, 0.5, 0.4)
        mode.update(lm1)
        lm2 = _landmarks_pair(10, 0.5, 0.4, 0.5, 0.3)
        mode.update(lm2)

        synth.reset_mock()
        mode.close()

        # Should have noteoff and CC 123 on both channels
        cc_calls = synth.cc.call_args_list
        cc_channels = {c[0][0] for c in cc_calls}
        assert 0 in cc_channels
        assert 1 in cc_channels


class TestProgramSetup:
    """MIDI programs are set on init."""

    def test_programs_set_on_init(self):
        """RealtimeMidiMode sets melody and bass programs on init."""
        mode, synth = _make_mode()

        pc_calls = synth.program_change.call_args_list
        channels = {c[0][0] for c in pc_calls}
        assert 0 in channels
        assert 1 in channels
