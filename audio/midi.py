from __future__ import annotations

import mido


class MidiOut:
    """Thin wrapper around a mido output port."""

    def __init__(self, port: mido.ports.BaseOutput) -> None:
        self._port = port

    def note_on(self, channel: int, note: int, velocity: int) -> None:
        velocity = max(0, min(127, velocity))
        self._port.send(mido.Message(
            "note_on", channel=channel, note=note, velocity=velocity,
        ))

    def note_off(self, channel: int, note: int) -> None:
        self._port.send(mido.Message(
            "note_off", channel=channel, note=note, velocity=0,
        ))

    def control_change(self, channel: int, control: int, value: int) -> None:
        self._port.send(mido.Message(
            "control_change", channel=channel, control=control, value=value,
        ))

    def program_change(self, channel: int, program: int) -> None:
        self._port.send(mido.Message(
            "program_change", channel=channel, program=program,
        ))

    def all_notes_off(self, channel: int) -> None:
        self.control_change(channel=channel, control=123, value=0)
