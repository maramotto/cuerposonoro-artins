"""Realtime MIDI mode — per-keypoint velocity-driven, D minor pentatonic.

Optimised for Jetson's 18 FPS camera and variable latency.
No beat grid: notes triggered purely by movement velocity with hysteresis.
Each tracked keypoint (wrists, ankles) maps to its own pitch range.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from vision.landmarks import Landmarks

log = logging.getLogger(__name__)

# D minor pentatonic pitch classes
_D_MINOR_PENTATONIC_PC = {0, 2, 5, 7, 9}  # C, D, F, G, A


def _scale_notes_in_range(note_min: int, note_max: int) -> list[int]:
    """Return all D minor pentatonic MIDI notes within [note_min, note_max]."""
    return [
        n for n in range(note_min, note_max + 1)
        if n % 12 in _D_MINOR_PENTATONIC_PC
    ]


class _KeypointTracker:
    """State machine for a single tracked keypoint."""

    __slots__ = (
        "index", "channel", "notes", "frames_above", "frames_below",
        "active_note", "last_noteon_time",
    )

    def __init__(self, index: int, channel: int, note_min: int, note_max: int) -> None:
        self.index = index
        self.channel = channel
        self.notes = _scale_notes_in_range(note_min, note_max)
        self.frames_above = 0
        self.frames_below = 0
        self.active_note: int | None = None
        self.last_noteon_time = -1e9  # ensure first note is never blocked by cooldown


class RealtimeMidiMode:
    """Per-keypoint velocity-driven MIDI — D minor pentatonic, no beat grid."""

    def __init__(self, synth, config: dict) -> None:
        rt = config["realtime"]
        self._synth = synth
        self._min_velocity = rt["min_velocity"]
        self._hysteresis_frames = rt["hysteresis_frames"]
        self._silence_frames = rt["silence_frames"]
        self._velocity_max = rt["velocity_max"]
        self._vel_min = rt["midi_velocity_min"]
        self._vel_max = rt["midi_velocity_max"]
        self._cooldown_s = rt.get("note_cooldown_ms", 0) / 1000.0

        self._trackers: list[_KeypointTracker] = []
        for _name, kp_cfg in rt["keypoints"].items():
            self._trackers.append(_KeypointTracker(
                index=kp_cfg["index"],
                channel=kp_cfg["channel"],
                note_min=kp_cfg["note_min"],
                note_max=kp_cfg["note_max"],
            ))

        # Set MIDI programs
        programs = rt["programs"]
        synth.program_change(
            programs["melody"]["channel"], programs["melody"]["program"],
        )
        synth.program_change(
            programs["bass"]["channel"], programs["bass"]["program"],
        )

    def update(self, landmarks: Landmarks) -> None:
        """Process one frame of landmarks and send MIDI as needed."""
        for tracker in self._trackers:
            vel = landmarks.velocity(tracker.index)
            self._update_tracker(tracker, vel)

    def _update_tracker(self, t: _KeypointTracker, velocity: float) -> None:
        if velocity > self._min_velocity:
            t.frames_above += 1
            t.frames_below = 0
        else:
            t.frames_above = 0
            t.frames_below += 1

        # Hysteresis: trigger note after N consecutive frames above threshold
        if t.frames_above >= self._hysteresis_frames:
            now = time.monotonic()
            if self._cooldown_s > 0 and (now - t.last_noteon_time) < self._cooldown_s:
                pass  # skip — cooldown active
            else:
                note = self._velocity_to_note(velocity, t.notes)
                midi_vel = self._velocity_to_midi_velocity(velocity)

                if note != t.active_note:
                    if t.active_note is not None:
                        self._synth.noteoff(t.channel, t.active_note)
                    self._synth.noteon(t.channel, note, midi_vel)
                    t.active_note = note
                    t.last_noteon_time = now

        # Silence: release note after N consecutive frames below threshold
        if t.frames_below >= self._silence_frames and t.active_note is not None:
            self._synth.noteoff(t.channel, t.active_note)
            t.active_note = None

    def _velocity_to_note(self, velocity: float, notes: list[int]) -> int:
        """Map movement velocity to a note in the pentatonic scale."""
        if not notes:
            return 60  # fallback (should never happen)
        t = np.clip(
            (velocity - self._min_velocity) / (self._velocity_max - self._min_velocity),
            0.0, 1.0,
        )
        idx = int(t * (len(notes) - 1))
        return notes[idx]

    def _velocity_to_midi_velocity(self, velocity: float) -> int:
        """Return maximum MIDI velocity (attack strength)."""
        return 127

    def close(self) -> None:
        """Release all held notes and send All Notes Off."""
        channels_seen: set[int] = set()
        for t in self._trackers:
            if t.active_note is not None:
                self._synth.noteoff(t.channel, t.active_note)
                t.active_note = None
            channels_seen.add(t.channel)

        for ch in channels_seen:
            self._synth.cc(ch, 123, 0)
