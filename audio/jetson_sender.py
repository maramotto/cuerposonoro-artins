"""Jetson MIDI sender — velocity-driven, sustained-note mode.

Designed for variable-latency, low-FPS pipelines (~18 FPS on Jetson).
Notes are triggered purely by movement velocity with hysteresis filtering.
No beat grid, no tempo assumptions.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from audio.chords import ChordProgression
from features.arms import ArmFeatures
from features.harmony import HarmonyFeatures
from features.legs import LegFeatures
from vision.landmarks import Landmarks

log = logging.getLogger(__name__)


class JetsonMidiSender:
    """Velocity-driven MIDI sender with hysteresis and sustained notes."""

    def __init__(self, synth, config: dict, chord_progression: ChordProgression) -> None:
        self._synth = synth
        self._mel = config["melody"]
        self._bass = config["bass"]
        self._harm = config["harmony"]
        self._silence_cfg = config["silence"]
        self._hysteresis_frames = config["jetson"]["hysteresis_frames"]
        self._cooldown_s = config["jetson"].get("note_cooldown_ms", 0) / 1000.0
        self._progression = chord_progression

        # State: currently held notes
        self._melody_note: int | None = None
        self._bass_note: int | None = None

        # Hysteresis counters
        self._arm_frames_above = 0
        self._ankle_frames_above = 0

        # Cooldown timestamps
        self._melody_last_noteon = -1e9  # ensure first note is never blocked
        self._bass_last_noteon = -1e9

        # Silence tracking
        self._still_since: float | None = None

        # Set MIDI programs
        synth.program_change(self._mel["channel"], self._mel["program"])
        synth.program_change(self._bass["channel"], self._bass["program"])

    def update(self, landmarks: Landmarks) -> None:
        """Process one frame of landmarks and send MIDI as needed."""
        arms = ArmFeatures(landmarks)
        legs = LegFeatures(landmarks)
        harmony = HarmonyFeatures(landmarks)

        arm_vel = arms.arm_velocity()
        ankle_vel = legs.ankle_velocity()

        if self._check_silence(max(arm_vel, ankle_vel)):
            return

        tilt_sign = self._update_harmony(harmony)
        chord = self._progression.current

        self._update_melody(arm_vel, arms, chord, tilt_sign)
        self._update_bass(ankle_vel, chord)

    def _check_silence(self, body_vel: float) -> bool:
        """Return True if body is still long enough to silence all notes."""
        if body_vel < self._silence_cfg["velocity_threshold"]:
            if self._still_since is None:
                self._still_since = time.monotonic()
            elapsed = time.monotonic() - self._still_since
            if elapsed >= self._silence_cfg["timeout_ms"] / 1000.0:
                self._release_all()
                return True
        else:
            self._still_since = None
        return False

    def _update_harmony(self, harmony: HarmonyFeatures) -> float:
        """Advance/retreat chord progression and return tilt sign."""
        if harmony.should_advance(self._harm["torso_tilt_threshold"]):
            self._progression.advance()
            log.info("Chord → %s", self._progression.current.name)
        elif harmony.should_retreat(self._harm["torso_tilt_threshold"]):
            self._progression.retreat()
            log.info("Chord → %s", self._progression.current.name)

        head_tilt = harmony.head_tilt()
        if head_tilt > self._harm["head_tilt_threshold"]:
            return 1.0
        elif head_tilt < -self._harm["head_tilt_threshold"]:
            return -1.0
        return 0.0

    def _update_melody(self, arm_vel: float, arms: ArmFeatures, chord, tilt_sign: float) -> None:
        """Trigger or sustain melody note based on arm velocity."""
        if arm_vel > self._mel["trigger_threshold"]:
            self._arm_frames_above += 1
        else:
            self._arm_frames_above = 0
            if self._melody_note is not None:
                self._synth.noteoff(self._mel["channel"], self._melody_note)
                self._melody_note = None

        if self._arm_frames_above >= self._hysteresis_frames:
            now = time.monotonic()
            if self._cooldown_s > 0 and (now - self._melody_last_noteon) < self._cooldown_s:
                return
            note = chord.note_from_height(
                arms.mean_wrist_height(),
                self._mel["note_min"],
                self._mel["note_max"],
                tilt=tilt_sign,
            )
            if note != self._melody_note:
                if self._melody_note is not None:
                    self._synth.noteoff(self._mel["channel"], self._melody_note)
                self._synth.noteon(self._mel["channel"], note, 127)
                self._melody_note = note
                self._melody_last_noteon = now

    def _update_bass(self, ankle_vel: float, chord) -> None:
        """Trigger or sustain bass note based on ankle velocity."""
        if ankle_vel > self._bass["trigger_threshold"]:
            self._ankle_frames_above += 1
        else:
            self._ankle_frames_above = 0
            if self._bass_note is not None:
                self._synth.noteoff(self._bass["channel"], self._bass_note)
                self._bass_note = None

        if self._ankle_frames_above >= self._hysteresis_frames:
            now = time.monotonic()
            if self._cooldown_s > 0 and (now - self._bass_last_noteon) < self._cooldown_s:
                return
            bass_note = chord.root
            if bass_note != self._bass_note:
                if self._bass_note is not None:
                    self._synth.noteoff(self._bass["channel"], self._bass_note)
                self._synth.noteon(
                    self._bass["channel"], bass_note, self._bass["velocity"],
                )
                self._bass_note = bass_note
                self._bass_last_noteon = now

    def close(self) -> None:
        """Release all held notes and send All Notes Off."""
        self._release_all()
        self._synth.cc(self._mel["channel"], 123, 0)
        self._synth.cc(self._bass["channel"], 123, 0)

    def _release_all(self) -> None:
        if self._melody_note is not None:
            self._synth.noteoff(self._mel["channel"], self._melody_note)
            self._melody_note = None
        if self._bass_note is not None:
            self._synth.noteoff(self._bass["channel"], self._bass_note)
            self._bass_note = None
        self._arm_frames_above = 0
        self._ankle_frames_above = 0
