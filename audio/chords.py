from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chord:
    name: str
    root: int
    notes: list[int]
    simplified: Optional[list[int]] = field(default=None)
    tension: Optional[list[int]] = field(default=None)

    def __post_init__(self) -> None:
        if self.simplified is None:
            self.simplified = list(self.notes)
        if self.tension is None:
            self.tension = list(self.notes)

    def active_notes(self, tilt: float = 0.0) -> list[int]:
        """Return chord tones based on head tilt.

        tilt > 0 → blend toward tension variant
        tilt < 0 → blend toward simplified variant
        tilt == 0 → base notes
        """
        if tilt > 0:
            return list(self.tension)
        elif tilt < 0:
            return list(self.simplified)
        return list(self.notes)

    def all_notes_in_range(self, low: int, high: int, tilt: float = 0.0) -> list[int]:
        """Return all chord tones (across octaves) within [low, high], sorted."""
        result = set()
        for note in self.active_notes(tilt):
            pitch_class = note % 12
            for octave_note in range(pitch_class, 128, 12):
                if low <= octave_note <= high:
                    result.add(octave_note)
        return sorted(result)

    def nearest_note(self, target: int, tilt: float = 0.0) -> int:
        """Return the chord tone nearest to target across all octaves."""
        candidates = self.all_notes_in_range(max(0, target - 12), min(127, target + 12), tilt)
        if not candidates:
            candidates = self.all_notes_in_range(0, 127, tilt)
        return min(candidates, key=lambda n: abs(n - target))

    def note_from_height(self, height: float, note_min: int, note_max: int, tilt: float = 0.0) -> int:
        """Map a normalised height [0, 1] to the nearest chord tone in range."""
        raw_midi = note_min + height * (note_max - note_min)
        return self.nearest_note(int(raw_midi), tilt)


@dataclass
class ChordProgression:
    chords: list[Chord]
    index: int = 0

    @property
    def current(self) -> Chord:
        return self.chords[self.index]

    def advance(self) -> None:
        self.index = (self.index + 1) % len(self.chords)

    def retreat(self) -> None:
        if self.index > 0:
            self.index -= 1

    @classmethod
    def from_config(cls, config: list[dict]) -> ChordProgression:
        chords = [
            Chord(
                name=c["name"],
                root=c["root"],
                notes=c["notes"],
                simplified=c.get("simplified"),
                tension=c.get("tension"),
            )
            for c in config
        ]
        return cls(chords=chords)
