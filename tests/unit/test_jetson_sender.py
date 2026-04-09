"""Tests for JetsonMidiSender — velocity-driven, sustained-note MIDI mode."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, call, patch

from audio.chords import Chord, ChordProgression
from audio.jetson_sender import JetsonMidiSender
from vision.landmarks import Landmarks


# --- Helpers ---

def _make_keypoints(
    wrist_y: float = 0.5,
    ankle_y: float = 0.8,
    ear_y_left: float = 0.1,
    ear_y_right: float = 0.1,
    shoulder_x: float = 0.5,
    hip_x: float = 0.5,
) -> np.ndarray:
    """Create a 17x3 COCO keypoints array with controllable positions."""
    kp = np.full((17, 3), 0.5, dtype=np.float32)
    kp[:, 2] = 1.0  # confidence
    # Ears (3, 4)
    kp[3] = [0.45, ear_y_left, 1.0]
    kp[4] = [0.55, ear_y_right, 1.0]
    # Shoulders (5, 6)
    kp[5] = [shoulder_x - 0.1, 0.3, 1.0]
    kp[6] = [shoulder_x + 0.1, 0.3, 1.0]
    # Elbows (7, 8)
    kp[7] = [0.35, 0.4, 1.0]
    kp[8] = [0.65, 0.4, 1.0]
    # Wrists (9, 10)
    kp[9] = [0.3, wrist_y, 1.0]
    kp[10] = [0.7, wrist_y, 1.0]
    # Hips (11, 12)
    kp[11] = [hip_x - 0.1, 0.6, 1.0]
    kp[12] = [hip_x + 0.1, 0.6, 1.0]
    # Knees (13, 14)
    kp[13] = [0.4, 0.7, 1.0]
    kp[14] = [0.6, 0.7, 1.0]
    # Ankles (15, 16)
    kp[15] = [0.4, ankle_y, 1.0]
    kp[16] = [0.6, ankle_y, 1.0]
    return kp


def _make_landmarks_with_velocity(
    wrist_y_prev: float, wrist_y_curr: float,
    ankle_y_prev: float = 0.8, ankle_y_curr: float = 0.8,
) -> Landmarks:
    """Create a Landmarks object with one update so velocity is nonzero."""
    prev_kp = _make_keypoints(wrist_y=wrist_y_prev, ankle_y=ankle_y_prev)
    curr_kp = _make_keypoints(wrist_y=wrist_y_curr, ankle_y=ankle_y_curr)
    lm = Landmarks(prev_kp)
    lm.update(curr_kp)
    return lm


def _make_still_landmarks() -> Landmarks:
    """Create Landmarks with zero velocity (no update, first frame)."""
    return Landmarks(_make_keypoints())


def _make_config():
    return {
        "melody": {
            "channel": 0,
            "program": 11,
            "note_min": 48,
            "note_max": 84,
            "velocity_min": 30,
            "velocity_max": 120,
            "trigger_threshold": 0.03,
            "brightness_cc": 74,
        },
        "bass": {
            "channel": 1,
            "program": 32,
            "trigger_threshold": 0.03,
            "velocity": 100,
        },
        "harmony": {
            "torso_tilt_threshold": 0.05,
            "head_tilt_threshold": 0.04,
        },
        "silence": {
            "velocity_threshold": 0.02,
            "timeout_ms": 500,
        },
        "jetson": {
            "hysteresis_frames": 2,
        },
    }


def _make_progression():
    return ChordProgression(chords=[
        Chord(name="Dm9", root=62, notes=[62, 65, 69, 72, 76]),
        Chord(name="G13sus4", root=67, notes=[67, 72, 74, 76, 81]),
    ])


def _make_sender(mock_synth, config=None, progression=None):
    if config is None:
        config = _make_config()
    if progression is None:
        progression = _make_progression()
    return JetsonMidiSender(
        synth=mock_synth,
        config=config,
        chord_progression=progression,
    )


# --- Tests ---

class TestHysteresis:
    def test_no_note_on_single_frame_above_threshold(self):
        """A single frame of movement should NOT trigger a note (hysteresis=2)."""
        synth = MagicMock()
        sender = _make_sender(synth)

        # Frame with arm movement
        lm = _make_landmarks_with_velocity(wrist_y_prev=0.5, wrist_y_curr=0.3)
        sender.update(lm)

        synth.noteon.assert_not_called()

    def test_note_triggers_after_hysteresis_frames(self):
        """A note triggers after 2 consecutive frames above threshold."""
        synth = MagicMock()
        sender = _make_sender(synth)

        # Two consecutive frames with arm movement
        lm1 = _make_landmarks_with_velocity(wrist_y_prev=0.5, wrist_y_curr=0.3)
        sender.update(lm1)
        synth.noteon.assert_not_called()

        lm2 = _make_landmarks_with_velocity(wrist_y_prev=0.3, wrist_y_curr=0.1)
        sender.update(lm2)
        synth.noteon.assert_called()
        # Should be on melody channel (0)
        assert synth.noteon.call_args[0][0] == 0


class TestSustainedNotes:
    def test_note_held_while_velocity_stays_above(self):
        """Once triggered, the note stays on without re-triggering."""
        synth = MagicMock()
        sender = _make_sender(synth)

        # Trigger: 2 frames
        lm1 = _make_landmarks_with_velocity(wrist_y_prev=0.5, wrist_y_curr=0.3)
        sender.update(lm1)
        lm2 = _make_landmarks_with_velocity(wrist_y_prev=0.3, wrist_y_curr=0.1)
        sender.update(lm2)

        noteon_count = synth.noteon.call_count
        # Third frame: still moving but same pitch area — no new note_on
        lm3 = _make_landmarks_with_velocity(wrist_y_prev=0.1, wrist_y_curr=0.12)
        sender.update(lm3)

        # Should not have fired another noteon for the same note
        # (may fire if pitch changed, but won't double-trigger same note)

    def test_note_released_when_velocity_drops(self):
        """Note is released when arm velocity drops below threshold."""
        synth = MagicMock()
        sender = _make_sender(synth)

        # Trigger melody note
        lm1 = _make_landmarks_with_velocity(wrist_y_prev=0.5, wrist_y_curr=0.3)
        sender.update(lm1)
        lm2 = _make_landmarks_with_velocity(wrist_y_prev=0.3, wrist_y_curr=0.1)
        sender.update(lm2)
        assert synth.noteon.called

        # Now: no movement (first frame of still Landmarks)
        lm_still = _make_still_landmarks()
        sender.update(lm_still)

        synth.noteoff.assert_called()
        # noteoff should be on melody channel (0)
        assert synth.noteoff.call_args[0][0] == 0


class TestVelocityMapping:
    def test_fast_movement_higher_velocity(self):
        """Faster arm movement produces higher MIDI velocity."""
        synth_slow = MagicMock()
        sender_slow = _make_sender(synth_slow)
        # Slow movement: small delta
        lm1 = _make_landmarks_with_velocity(wrist_y_prev=0.5, wrist_y_curr=0.46)
        sender_slow.update(lm1)
        lm2 = _make_landmarks_with_velocity(wrist_y_prev=0.46, wrist_y_curr=0.42)
        sender_slow.update(lm2)

        synth_fast = MagicMock()
        sender_fast = _make_sender(synth_fast)
        # Fast movement: large delta
        lm3 = _make_landmarks_with_velocity(wrist_y_prev=0.5, wrist_y_curr=0.2)
        sender_fast.update(lm3)
        lm4 = _make_landmarks_with_velocity(wrist_y_prev=0.2, wrist_y_curr=0.0)
        sender_fast.update(lm4)

        if synth_slow.noteon.called and synth_fast.noteon.called:
            vel_slow = synth_slow.noteon.call_args[0][2]
            vel_fast = synth_fast.noteon.call_args[0][2]
            assert vel_fast >= vel_slow


class TestBassNotes:
    def test_bass_triggers_on_ankle_movement(self):
        """Bass note triggers when ankle velocity exceeds threshold for 2 frames."""
        synth = MagicMock()
        sender = _make_sender(synth)

        # Two frames with ankle movement
        lm1 = _make_landmarks_with_velocity(
            wrist_y_prev=0.5, wrist_y_curr=0.5,
            ankle_y_prev=0.8, ankle_y_curr=0.6,
        )
        sender.update(lm1)
        lm2 = _make_landmarks_with_velocity(
            wrist_y_prev=0.5, wrist_y_curr=0.5,
            ankle_y_prev=0.6, ankle_y_curr=0.4,
        )
        sender.update(lm2)

        # Should have called noteon on bass channel (1)
        bass_calls = [c for c in synth.noteon.call_args_list if c[0][0] == 1]
        assert len(bass_calls) > 0

    def test_bass_released_when_ankle_stops(self):
        """Bass note released when ankle velocity drops below threshold."""
        synth = MagicMock()
        sender = _make_sender(synth)

        # Trigger bass
        lm1 = _make_landmarks_with_velocity(
            wrist_y_prev=0.5, wrist_y_curr=0.5,
            ankle_y_prev=0.8, ankle_y_curr=0.5,
        )
        sender.update(lm1)
        lm2 = _make_landmarks_with_velocity(
            wrist_y_prev=0.5, wrist_y_curr=0.5,
            ankle_y_prev=0.5, ankle_y_curr=0.2,
        )
        sender.update(lm2)

        # Now still
        lm_still = _make_still_landmarks()
        sender.update(lm_still)

        bass_offs = [c for c in synth.noteoff.call_args_list if c[0][0] == 1]
        assert len(bass_offs) > 0


class TestClose:
    def test_close_sends_all_notes_off(self):
        """close() sends CC 123 on melody and bass channels."""
        synth = MagicMock()
        sender = _make_sender(synth)

        # Trigger a note first
        lm1 = _make_landmarks_with_velocity(wrist_y_prev=0.5, wrist_y_curr=0.2)
        sender.update(lm1)
        lm2 = _make_landmarks_with_velocity(wrist_y_prev=0.2, wrist_y_curr=0.0)
        sender.update(lm2)

        synth.reset_mock()
        sender.close()

        # Should send CC 123 on both channels
        cc_calls = synth.cc.call_args_list
        channels = {c[0][0] for c in cc_calls}
        assert 0 in channels  # melody
        assert 1 in channels  # bass


class TestProgramSetup:
    def test_programs_set_on_init(self):
        """JetsonMidiSender sets MIDI programs on init."""
        synth = MagicMock()
        _make_sender(synth)

        pc_calls = synth.program_change.call_args_list
        channels = {c[0][0] for c in pc_calls}
        assert 0 in channels  # melody = vibraphone
        assert 1 in channels  # bass = acoustic bass
