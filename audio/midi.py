from __future__ import annotations


class MidiOut:
    """Thin wrapper around a pyfluidsynth Synth object."""

    def __init__(self, synth) -> None:
        self._synth = synth

    def note_on(self, channel: int, note: int, velocity: int) -> None:
        velocity = max(0, min(127, velocity))
        self._synth.noteon(channel, note, velocity)

    def note_off(self, channel: int, note: int) -> None:
        self._synth.noteoff(channel, note)

    def control_change(self, channel: int, control: int, value: int) -> None:
        self._synth.cc(channel, control, value)

    def program_change(self, channel: int, program: int) -> None:
        self._synth.program_change(channel, program)

    def all_notes_off(self, channel: int) -> None:
        self.control_change(channel=channel, control=123, value=0)
