"""Gesture MIDI mode — direction-based, 3 voices, Re dorian.

Maps body movement *direction* to musical phrases: upward gesture = ascending
notes, downward = descending. Energy (global velocity) controls volume (CC7).
Duration determines articulation: short gestures = staccato, long = reverb tail.

Designed for Jetson's 18 FPS camera and variable latency.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, NamedTuple

import numpy as np

from vision.landmarks import Landmarks

log = logging.getLogger(__name__)


class _DeferredAction(NamedTuple):
    """A MIDI action scheduled to execute after a number of frames."""
    frames_remaining: int
    action: Callable[[], None]


# D dorian pitch classes: D E F G A B C
_D_DORIAN_PC = {0, 2, 4, 5, 7, 9, 11}


def _dorian_notes_in_range(note_min: int, note_max: int) -> list[int]:
    """Return all D dorian MIDI notes within [note_min, note_max]."""
    return [
        n for n in range(note_min, note_max + 1)
        if n % 12 in _D_DORIAN_PC
    ]


class _VoiceTracker:
    """State machine for one voice (a group of keypoints)."""

    __slots__ = (
        "name", "indices", "channel", "notes", "note_index",
        "active_note", "direction_count", "last_direction",
        "frames_active", "frames_silent", "last_noteon_time",
    )

    def __init__(
        self, name: str, indices: list[int], channel: int,
        note_min: int, note_max: int,
    ) -> None:
        self.name = name
        self.indices = indices
        self.channel = channel
        self.notes = _dorian_notes_in_range(note_min, note_max)
        self.note_index = len(self.notes) // 2  # start mid-scale
        self.active_note: int | None = None
        self.direction_count = 0
        self.last_direction = 0  # -1 = up, +1 = down, 0 = none
        self.frames_active = 0
        self.frames_silent = 0
        self.last_noteon_time = -1e9  # ensure first note is never blocked by cooldown


class GestureMidiMode:
    """Direction-based MIDI with 3 voices, Re dorian, energy + articulation."""

    # Voice definitions: (name, keypoint indices, channel, note_min, note_max)
    _VOICE_DEFS = [
        ("right_arm", [10, 8], 0, 74, 84),   # D5–C6
        ("left_arm", [9, 7], 0, 62, 72),      # D4–C5
        ("center", [11, 12], 1, 38, 43),       # D2–G2
    ]

    def __init__(self, synth, config: dict) -> None:
        gc = config["gesture"]
        self._synth = synth
        self._min_velocity = gc["min_velocity"]
        self._hysteresis_frames = gc["hysteresis_frames"]
        self._silence_frames = gc["silence_frames"]
        self._velocity_max = gc["velocity_max"]
        self._staccato_frames = gc["staccato_frames"]
        self._sustain_frames = gc["sustain_frames"]
        self._reverb_cc91 = gc["reverb_cc91"]
        self._cooldown_s = gc.get("note_cooldown_ms", 0) / 1000.0

        self._voices = [
            _VoiceTracker(name, indices, ch, nmin, nmax)
            for name, indices, ch, nmin, nmax in self._VOICE_DEFS
        ]

        self._deferred: list[_DeferredAction] = []

        # Set MIDI programs
        programs = gc["programs"]
        synth.program_change(
            programs["arms"]["channel"], programs["arms"]["program"],
        )
        synth.program_change(
            programs["center"]["channel"], programs["center"]["program"],
        )

    def update(self, landmarks: Landmarks) -> None:
        """Process one frame of landmarks."""
        # Tick deferred actions
        self._tick_deferred()

        # Update each voice
        for voice in self._voices:
            self._update_voice(voice, landmarks)

        # Global energy → CC7
        self._update_energy(landmarks)

    def _update_voice(self, v: _VoiceTracker, landmarks: Landmarks) -> None:
        # Compute mean velocity and mean delta_y for this voice's keypoints
        velocities = [landmarks.velocity(i) for i in v.indices]
        mean_vel = float(np.mean(velocities))

        if mean_vel >= self._min_velocity:
            v.frames_active += 1
            v.frames_silent = 0

            # Compute direction from delta_y of mean position
            direction = self._compute_direction(v, landmarks)
            self._update_hysteresis(v, direction, mean_vel)
        else:
            v.frames_silent += 1

            if v.frames_silent >= self._silence_frames and v.active_note is not None:
                self._release_note(v)

            v.direction_count = 0
            v.last_direction = 0

    def _compute_direction(self, v: _VoiceTracker, landmarks: Landmarks) -> int:
        """Compute vertical direction from landmark deltas. Returns -1 (up), +1 (down), 0."""
        if landmarks._prev is None:
            return 0

        deltas_y = []
        for i in v.indices:
            curr_y = float(landmarks.keypoints[i, 1])
            prev_y = float(landmarks._prev[i, 1])
            deltas_y.append(curr_y - prev_y)

        mean_dy = float(np.mean(deltas_y))

        if mean_dy < -self._min_velocity:
            return -1  # moving UP (y decreases in image coords)
        elif mean_dy > self._min_velocity:
            return 1   # moving DOWN
        return 0

    def _update_hysteresis(self, v: _VoiceTracker, direction: int, velocity: float) -> None:
        if direction == 0:
            return

        if direction == v.last_direction:
            v.direction_count += 1
        else:
            v.direction_count = 1
            v.last_direction = direction

        if v.direction_count >= self._hysteresis_frames:
            now = time.monotonic()
            if self._cooldown_s > 0 and (now - v.last_noteon_time) < self._cooldown_s:
                v.direction_count = 0
                return

            # Determine step count from velocity magnitude (1–3)
            t = np.clip(
                (velocity - self._min_velocity) / (self._velocity_max - self._min_velocity),
                0.0, 1.0,
            )
            steps = 1 + int(t * 2)  # 1, 2, or 3

            # direction=-1 means UP in image → ascending notes (higher index)
            new_index = v.note_index - (direction * steps)
            new_index = max(0, min(len(v.notes) - 1, new_index))

            if new_index != v.note_index or v.active_note is None:
                v.note_index = new_index
                new_note = v.notes[v.note_index]

                if new_note != v.active_note:
                    if v.active_note is not None:
                        self._synth.noteoff(v.channel, v.active_note)
                    midi_vel = self._velocity_to_midi(velocity)
                    self._synth.noteon(v.channel, new_note, midi_vel)
                    v.active_note = new_note
                    v.last_noteon_time = now

            # Reset hysteresis counter after triggering
            v.direction_count = 0

    def _release_note(self, v: _VoiceTracker) -> None:
        """Release a voice's note with articulation logic."""
        if v.active_note is None:
            return

        note = v.active_note
        channel = v.channel

        if v.frames_active >= self._sustain_frames:
            # Sustained: add reverb before releasing
            self._synth.cc(channel, 91, self._reverb_cc91)
            # Defer noteoff by 2 frames
            self._deferred.append(_DeferredAction(2, lambda ch=channel, n=note: self._synth.noteoff(ch, n)))
        else:
            # Staccato: immediate release
            self._synth.noteoff(channel, note)

        v.active_note = None
        v.frames_active = 0

    def _update_energy(self, landmarks: Landmarks) -> None:
        """Set CC7 (volume) to maximum on both channels."""
        channels_seen: set[int] = set()
        for v in self._voices:
            if v.channel not in channels_seen:
                self._synth.cc(v.channel, 7, 127)
                channels_seen.add(v.channel)

    def _velocity_to_midi(self, velocity: float) -> int:
        """Return maximum MIDI note velocity."""
        return 127

    def _tick_deferred(self) -> None:
        """Decrement deferred action counters and execute when ready."""
        remaining = []
        for entry in self._deferred:
            if entry.frames_remaining <= 1:
                entry.action()
            else:
                remaining.append(_DeferredAction(entry.frames_remaining - 1, entry.action))
        self._deferred = remaining

    def close(self) -> None:
        """Release all held notes and send All Notes Off."""
        channels_seen: set[int] = set()
        for v in self._voices:
            if v.active_note is not None:
                self._synth.noteoff(v.channel, v.active_note)
                v.active_note = None
            channels_seen.add(v.channel)

        # Execute any deferred actions
        for entry in self._deferred:
            entry.action()
        self._deferred.clear()

        for ch in channels_seen:
            self._synth.cc(ch, 123, 0)
