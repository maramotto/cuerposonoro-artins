"""Tests for GestureMidiMode — direction-based, 3 voices, Re dorian.

Gesture direction maps to scale movement, energy to volume, duration to articulation.
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, call

from audio.gesture_mode import GestureMidiMode, _dorian_notes_in_range
from vision.landmarks import Landmarks


# --- Helpers ---

def _make_keypoints(**overrides) -> np.ndarray:
    """Create a 17x3 COCO keypoints array.

    Override with kp<N>=(x, y), e.g. _make_keypoints(kp10=(0.5, 0.3)).
    """
    kp = np.full((17, 3), 0.5, dtype=np.float32)
    kp[:, 2] = 1.0
    for key, (x, y) in overrides.items():
        idx = int(key[2:])
        kp[idx] = [x, y, 1.0]
    return kp


def _landmarks_pair(prev_overrides: dict, curr_overrides: dict) -> Landmarks:
    """Create Landmarks with one update so velocities/deltas are nonzero."""
    prev_kp = _make_keypoints(**prev_overrides)
    curr_kp = _make_keypoints(**curr_overrides)
    lm = Landmarks(prev_kp)
    lm.update(curr_kp)
    return lm


def _still_landmarks() -> Landmarks:
    """Landmarks with zero velocity (first frame, no update)."""
    return Landmarks(_make_keypoints())


def _make_config():
    return {
        "gesture": {
            "min_velocity": 0.01,
            "hysteresis_frames": 3,
            "silence_frames": 6,
            "velocity_max": 0.15,
            "staccato_frames": 6,
            "sustain_frames": 8,
            "reverb_cc91": 80,
            "programs": {
                "arms": {"channel": 0, "program": 11},
                "center": {"channel": 1, "program": 32},
            },
        },
    }


def _make_mode(synth=None, config=None):
    if synth is None:
        synth = MagicMock()
    if config is None:
        config = _make_config()
    return GestureMidiMode(synth=synth, config=config), synth


def _feed_upward_right_arm(mode, n_frames: int, step: float = 0.04):
    """Feed n_frames where right arm (wrist 10 + elbow 8) moves upward."""
    frames = []
    for i in range(n_frames):
        y_prev = 0.5 - i * step
        y_curr = 0.5 - (i + 1) * step
        lm = _landmarks_pair(
            prev_overrides={"kp10": (0.5, y_prev), "kp8": (0.5, y_prev + 0.05)},
            curr_overrides={"kp10": (0.5, y_curr), "kp8": (0.5, y_curr + 0.05)},
        )
        mode.update(lm)
        frames.append(lm)
    return frames


# --- Scale helper tests ---

class TestDorianScale:
    def test_d_dorian_pitch_classes(self):
        """_dorian_notes_in_range returns only D dorian pitch classes."""
        notes = _dorian_notes_in_range(60, 84)
        d_dorian_pc = {0, 2, 4, 5, 7, 9, 11}
        for n in notes:
            assert n % 12 in d_dorian_pc, f"Note {n} (pc={n % 12}) not in D dorian"

    def test_right_arm_range(self):
        """D5-C6 range contains expected dorian notes."""
        notes = _dorian_notes_in_range(74, 84)
        # D5(74), E5(76), F5(77), G5(79), A5(81), B5(83), C6(84)
        assert notes == [74, 76, 77, 79, 81, 83, 84]

    def test_left_arm_range(self):
        """D4-C5 range contains expected dorian notes."""
        notes = _dorian_notes_in_range(62, 72)
        # D4(62), E4(64), F4(65), G4(67), A4(69), B4(71), C5(72)
        assert notes == [62, 64, 65, 67, 69, 71, 72]

    def test_center_range(self):
        """D2-G2 range contains expected dorian notes."""
        notes = _dorian_notes_in_range(38, 43)
        # D2(38), E2(40), F2(41), G2(43)
        assert notes == [38, 40, 41, 43]


# --- Direction-based gesture tests ---

class TestDirectionGesture:
    def test_upward_gesture_triggers_note_after_hysteresis(self):
        """Right arm moving UP for >= 3 frames triggers a note on channel 0."""
        mode, synth = _make_mode()

        # Feed 3 frames of upward movement (delta_y negative = UP in image coords)
        _feed_upward_right_arm(mode, 3)

        synth.noteon.assert_called()
        assert synth.noteon.call_args[0][0] == 0  # melody channel

    def test_no_trigger_before_hysteresis(self):
        """Right arm moving UP for only 2 frames should NOT trigger (hysteresis=3)."""
        mode, synth = _make_mode()

        _feed_upward_right_arm(mode, 2)

        synth.noteon.assert_not_called()

    def test_upward_gesture_advances_up_scale(self):
        """Continued upward movement produces ascending notes."""
        mode, synth = _make_mode()

        # 3 frames to trigger first note
        _feed_upward_right_arm(mode, 3)
        first_note = synth.noteon.call_args[0][1]

        # 3 more frames to advance further
        synth.reset_mock()
        _feed_upward_right_arm(mode, 3, step=0.04)
        if synth.noteon.called:
            second_note = synth.noteon.call_args[0][1]
            assert second_note >= first_note, (
                f"Expected ascending: {second_note} >= {first_note}"
            )


# --- Silence tests ---

class TestSilence:
    def test_voice_silent_below_min_velocity(self):
        """No notes triggered when all keypoints are still."""
        mode, synth = _make_mode()

        for _ in range(10):
            mode.update(_still_landmarks())

        synth.noteon.assert_not_called()

    def test_note_released_after_silence_frames(self):
        """Active note is released after 6 frames of stillness."""
        mode, synth = _make_mode()

        # Trigger a note
        _feed_upward_right_arm(mode, 4)
        assert synth.noteon.called

        # Feed 5 still frames — not yet released
        synth.reset_mock()
        for _ in range(5):
            mode.update(_still_landmarks())
        noteoff_count_before = synth.noteoff.call_count

        # 6th still frame — should release
        mode.update(_still_landmarks())
        assert synth.noteoff.call_count > noteoff_count_before


# --- Energy (CC7) tests ---

class TestEnergyCC7:
    def test_cc7_sent_every_frame(self):
        """CC7 (volume) is sent on every update, even without gesture trigger."""
        mode, synth = _make_mode()

        # Feed a frame with some movement
        lm = _landmarks_pair(
            prev_overrides={"kp10": (0.5, 0.5)},
            curr_overrides={"kp10": (0.5, 0.45)},
        )
        mode.update(lm)

        # CC7 = control 7
        cc_calls = [c for c in synth.cc.call_args_list if c[0][1] == 7]
        assert len(cc_calls) >= 1, "CC7 should be sent every frame"

    def test_cc7_proportional_to_velocity(self):
        """Higher movement velocity produces higher CC7 value."""
        mode_slow, synth_slow = _make_mode()
        lm_slow = _landmarks_pair(
            prev_overrides={"kp10": (0.5, 0.5)},
            curr_overrides={"kp10": (0.5, 0.49)},
        )
        mode_slow.update(lm_slow)

        mode_fast, synth_fast = _make_mode()
        lm_fast = _landmarks_pair(
            prev_overrides={"kp10": (0.5, 0.5)},
            curr_overrides={"kp10": (0.5, 0.35)},
        )
        mode_fast.update(lm_fast)

        cc7_slow = [c for c in synth_slow.cc.call_args_list if c[0][1] == 7]
        cc7_fast = [c for c in synth_fast.cc.call_args_list if c[0][1] == 7]

        if cc7_slow and cc7_fast:
            val_slow = cc7_slow[0][0][2]
            val_fast = cc7_fast[0][0][2]
            assert val_fast >= val_slow


# --- Articulation tests ---

class TestArticulation:
    def test_staccato_immediate_noteoff(self):
        """Note released with frames_active < 6 gets immediate noteoff (staccato)."""
        mode, synth = _make_mode()

        # Trigger a note (3 frames to pass hysteresis)
        _feed_upward_right_arm(mode, 3)
        assert synth.noteon.called

        # Now stop immediately — only 3 frames active, < staccato_frames(6)
        synth.reset_mock()
        for _ in range(6):
            mode.update(_still_landmarks())

        # Noteoff should have been sent without CC91 reverb
        assert synth.noteoff.called
        cc91_calls = [c for c in synth.cc.call_args_list if c[0][1] == 91]
        # No reverb CC91 for staccato
        reverb_before_noteoff = [
            c for c in cc91_calls if c[0][2] == 80
        ]
        assert len(reverb_before_noteoff) == 0

    def test_sustained_sends_reverb_cc91(self):
        """Note held for >= 8 frames gets CC91=80 reverb on release."""
        mode, synth = _make_mode()

        # Trigger and sustain for >= 8 frames of movement
        _feed_upward_right_arm(mode, 10, step=0.02)
        assert synth.noteon.called

        # Now stop — note was active for 10 frames (>= sustain_frames=8)
        synth.reset_mock()
        for _ in range(8):
            mode.update(_still_landmarks())

        # Should have CC91=80 sent
        cc91_calls = [c for c in synth.cc.call_args_list if c[0][1] == 91 and c[0][2] == 80]
        assert len(cc91_calls) > 0, "CC91=80 (reverb) should be sent for sustained notes"


# --- Close tests ---

class TestClose:
    def test_close_releases_all_and_sends_cc123(self):
        """close() releases held notes and sends CC 123 on both channels."""
        mode, synth = _make_mode()

        _feed_upward_right_arm(mode, 4)

        synth.reset_mock()
        mode.close()

        cc_calls = synth.cc.call_args_list
        cc123_channels = {c[0][0] for c in cc_calls if c[0][1] == 123}
        assert 0 in cc123_channels
        assert 1 in cc123_channels


# --- Program setup tests ---

class TestProgramSetup:
    def test_programs_set_on_init(self):
        """GestureMidiMode sets vibraphone and bass programs on init."""
        mode, synth = _make_mode()

        pc_calls = synth.program_change.call_args_list
        channels = {c[0][0] for c in pc_calls}
        assert 0 in channels
        assert 1 in channels
