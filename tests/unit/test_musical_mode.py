"""Tests for MusicalMidiMode — note triggers, cooldown, silence, harmony."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from audio.chords import ChordProgression
from vision.landmarks import Landmarks


def _config(cooldown_ms: int = 150) -> dict:
    return {
        "melody": {
            "channel": 0,
            "program": 11,
            "note_min": 48,
            "note_max": 84,
            "velocity_min": 30,
            "velocity_max": 120,
            "trigger_threshold": 0.018,
            "brightness_cc": 74,
            "note_cooldown_ms": cooldown_ms,
        },
        "bass": {
            "channel": 1,
            "program": 32,
            "trigger_threshold": 0.018,
            "velocity": 100,
        },
        "harmony": {
            "torso_tilt_threshold": 0.05,
            "head_tilt_threshold": 0.04,
        },
        "silence": {
            "velocity_threshold": 0.008,
            "timeout_ms": 500,
        },
    }


def _chord_progression() -> ChordProgression:
    chords = [
        {
            "name": "Dm9",
            "root": 62,
            "notes": [62, 65, 69, 72, 76],
            "simplified": [62, 65, 69],
            "tension": [62, 65, 69, 72, 76, 77],
        },
        {
            "name": "G13sus4",
            "root": 67,
            "notes": [67, 72, 74, 76, 81],
            "simplified": [67, 72, 74],
            "tension": [67, 72, 74, 76, 81, 83],
        },
    ]
    return ChordProgression.from_config(chords)


def _moving_landmarks(
    arm_dx: float = 0.1, ankle_dx: float = 0.0, wrist_y: float = 0.5,
) -> Landmarks:
    """Create landmarks with arm and/or ankle movement."""
    kp1 = np.zeros((17, 3), dtype=np.float32)
    kp1[:, 2] = 1.0
    kp1[5] = [0.3, 0.3, 1.0]
    kp1[6] = [0.7, 0.3, 1.0]
    kp1[7] = [0.2, 0.4, 1.0]
    kp1[8] = [0.8, 0.4, 1.0]
    kp1[9] = [0.3, wrist_y, 1.0]
    kp1[10] = [0.7, wrist_y, 1.0]
    kp1[11] = [0.4, 0.6, 1.0]
    kp1[12] = [0.6, 0.6, 1.0]
    kp1[15] = [0.4, 0.9, 1.0]
    kp1[16] = [0.6, 0.9, 1.0]

    kp2 = kp1.copy()
    for i in [5, 6, 7, 8, 9, 10]:
        kp2[i, 0] += arm_dx
    for i in [15, 16]:
        kp2[i, 0] += ankle_dx

    lm = Landmarks(kp1)
    lm.update(kp2)
    return lm


def _still_landmarks() -> Landmarks:
    """Create landmarks with no movement (below any threshold)."""
    kp1 = np.zeros((17, 3), dtype=np.float32)
    kp1[:, 2] = 1.0
    kp2 = kp1.copy()
    kp2[9, 0] += 0.001  # tiny jitter

    lm = Landmarks(kp1)
    lm.update(kp2)
    return lm


class TestMusicalModeNotes:
    def test_melody_triggers_on_arm_movement(self):
        from audio.musical_mode import MusicalMidiMode

        synth = MagicMock()
        mode = MusicalMidiMode(synth, _config(cooldown_ms=0), _chord_progression())

        mode.update(_moving_landmarks(arm_dx=0.1))

        assert synth.noteon.called
        # Melody is on channel 0
        melody_calls = [c for c in synth.noteon.call_args_list if c[0][0] == 0]
        assert len(melody_calls) > 0

    def test_bass_triggers_on_ankle_movement(self):
        from audio.musical_mode import MusicalMidiMode

        synth = MagicMock()
        mode = MusicalMidiMode(synth, _config(cooldown_ms=0), _chord_progression())

        mode.update(_moving_landmarks(arm_dx=0.0, ankle_dx=0.1))

        bass_calls = [c for c in synth.noteon.call_args_list if c[0][0] == 1]
        assert len(bass_calls) > 0

    def test_brightness_cc_sent_on_arm_movement(self):
        from audio.musical_mode import MusicalMidiMode

        synth = MagicMock()
        mode = MusicalMidiMode(synth, _config(cooldown_ms=0), _chord_progression())

        mode.update(_moving_landmarks(arm_dx=0.1))

        # CC74 (brightness) should be sent on channel 0
        cc_calls = [c for c in synth.cc.call_args_list if c[0][0] == 0 and c[0][1] == 74]
        assert len(cc_calls) > 0


class TestMusicalModeCooldown:
    def test_melody_rate_limited_by_cooldown(self):
        from audio.musical_mode import MusicalMidiMode

        synth = MagicMock()
        mode = MusicalMidiMode(synth, _config(cooldown_ms=500), _chord_progression())

        mode.update(_moving_landmarks(arm_dx=0.1))
        first_count = synth.noteon.call_count

        mode.update(_moving_landmarks(arm_dx=-0.1))

        # Should not trigger additional melody note during cooldown
        melody_calls = [c for c in synth.noteon.call_args_list if c[0][0] == 0]
        assert len(melody_calls) <= 1

    def test_note_triggers_after_cooldown_expires(self):
        from audio.musical_mode import MusicalMidiMode

        synth = MagicMock()
        mode = MusicalMidiMode(synth, _config(cooldown_ms=50), _chord_progression())

        mode.update(_moving_landmarks(arm_dx=0.1, wrist_y=0.5))
        first_count = synth.noteon.call_count

        time.sleep(0.06)

        mode.update(_moving_landmarks(arm_dx=0.1, wrist_y=0.2))

        assert synth.noteon.call_count > first_count


class TestMusicalModeSilence:
    def test_silence_releases_notes(self):
        from audio.musical_mode import MusicalMidiMode

        synth = MagicMock()
        mode = MusicalMidiMode(synth, _config(cooldown_ms=0), _chord_progression())

        # Trigger a melody note
        mode.update(_moving_landmarks(arm_dx=0.1))
        assert synth.noteon.called

        # Silence detection needs timeout — use zero timeout config
        cfg = _config(cooldown_ms=0)
        cfg["silence"]["timeout_ms"] = 0
        mode2 = MusicalMidiMode(synth, cfg, _chord_progression())
        synth.reset_mock()

        mode2.update(_moving_landmarks(arm_dx=0.1))
        assert synth.noteon.called

        synth.reset_mock()
        mode2.update(_still_landmarks())
        # Should release with noteoff
        assert synth.noteoff.called


class TestMusicalModeClose:
    def test_close_sends_all_notes_off(self):
        from audio.musical_mode import MusicalMidiMode

        synth = MagicMock()
        mode = MusicalMidiMode(synth, _config(cooldown_ms=0), _chord_progression())

        mode.update(_moving_landmarks(arm_dx=0.1))
        mode.close()

        # All Notes Off (CC 123) should be sent on both channels
        cc123_calls = [c for c in synth.cc.call_args_list if c[0][1] == 123]
        channels_off = {c[0][0] for c in cc123_calls}
        assert 0 in channels_off
        assert 1 in channels_off
