"""Musical MIDI mode — per-frame note triggers with chord progression.

Original cuerposonoro-jetson mode: melody from arms, bass from legs, harmony
from torso/head tilt. Notes are always consonant with the active chord.
"""
from __future__ import annotations

import logging
import time

from audio.chords import ChordProgression
from features.arms import ArmFeatures
from features.harmony import HarmonyFeatures
from features.legs import LegFeatures
from features.silence import SilenceTracker
from vision.landmarks import Landmarks

log = logging.getLogger(__name__)


class MusicalMidiMode:
    """Per-frame MIDI note triggers driven by arm/leg velocity and chord harmony."""

    def __init__(self, synth, config: dict, chord_progression: ChordProgression) -> None:
        self._synth = synth
        self._mel = config["melody"]
        self._bass = config["bass"]
        self._harm = config["harmony"]
        self._progression = chord_progression

        self._silence = SilenceTracker(
            threshold=config["silence"]["velocity_threshold"],
            timeout_ms=config["silence"]["timeout_ms"],
        )

        cooldown_ms = config.get("melody", {}).get("note_cooldown_ms", 150)
        self._cooldown_s = cooldown_ms / 1000.0
        self._last_melody_noteon = -1e9  # ensure first note is never blocked
        self._last_bass_noteon = -1e9

        # Currently held notes
        self._melody_note: int | None = None
        self._bass_note: int | None = None

        # Set MIDI programs
        synth.program_change(self._mel["channel"], self._mel["program"])
        synth.program_change(self._bass["channel"], self._bass["program"])

    def update(self, landmarks: Landmarks) -> None:
        """Process one frame of landmarks and send MIDI as needed."""
        body_vel = landmarks.mean_velocity(list(range(17)))
        is_silent = self._silence.update(body_vel)

        if is_silent:
            self._release_all()
            return

        self._update_harmony(landmarks)

        chord = self._progression.current
        head_tilt = HarmonyFeatures(landmarks).head_tilt()
        tilt_sign = self._tilt_sign(head_tilt)

        self._update_melody(landmarks, chord, tilt_sign)
        self._update_bass(landmarks, chord)

    def _update_harmony(self, landmarks: Landmarks) -> None:
        harmony = HarmonyFeatures(landmarks)
        threshold = self._harm["torso_tilt_threshold"]

        if harmony.should_advance(threshold):
            self._progression.advance()
            log.info("Chord → %s", self._progression.current.name)
        elif harmony.should_retreat(threshold):
            self._progression.retreat()
            log.info("Chord → %s", self._progression.current.name)

    def _update_melody(self, landmarks: Landmarks, chord, tilt_sign: float) -> None:
        arms = ArmFeatures(landmarks)
        arm_vel = arms.arm_velocity()
        now = time.monotonic()

        if arm_vel > self._mel["trigger_threshold"]:
            if self._cooldown_s <= 0 or (now - self._last_melody_noteon) >= self._cooldown_s:
                if self._melody_note is not None:
                    self._synth.noteoff(self._mel["channel"], self._melody_note)

                note = chord.note_from_height(
                    arms.mean_wrist_height(),
                    self._mel["note_min"],
                    self._mel["note_max"],
                    tilt=tilt_sign,
                )
                self._synth.noteon(self._mel["channel"], note, 127)
                self._melody_note = note
                self._last_melody_noteon = now

        self._synth.cc(
            self._mel["channel"], self._mel["brightness_cc"], arms.brightness(),
        )

    def _update_bass(self, landmarks: Landmarks, chord) -> None:
        legs = LegFeatures(landmarks)
        ankle_vel = legs.ankle_velocity()
        now = time.monotonic()

        if ankle_vel > self._bass["trigger_threshold"]:
            if self._cooldown_s <= 0 or (now - self._last_bass_noteon) >= self._cooldown_s:
                if self._bass_note is not None:
                    self._synth.noteoff(self._bass["channel"], self._bass_note)

                bass_note = chord.root
                self._synth.noteon(self._bass["channel"], bass_note, self._bass["velocity"])
                self._bass_note = bass_note
                self._last_bass_noteon = now

    def _tilt_sign(self, head_tilt: float) -> float:
        threshold = self._harm["head_tilt_threshold"]
        if head_tilt > threshold:
            return 1.0
        elif head_tilt < -threshold:
            return -1.0
        return 0.0

    def _release_all(self) -> None:
        if self._melody_note is not None:
            self._synth.noteoff(self._mel["channel"], self._melody_note)
            self._melody_note = None
        if self._bass_note is not None:
            self._synth.noteoff(self._bass["channel"], self._bass_note)
            self._bass_note = None

    def close(self) -> None:
        """Release all held notes and send All Notes Off."""
        self._release_all()
        self._synth.cc(self._mel["channel"], 123, 0)
        self._synth.cc(self._bass["channel"], 123, 0)
