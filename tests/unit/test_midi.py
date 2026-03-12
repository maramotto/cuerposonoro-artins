import pytest
from unittest.mock import MagicMock, call
from audio.midi import MidiOut


class TestMidiOut:
    @pytest.fixture
    def mock_port(self):
        return MagicMock()

    @pytest.fixture
    def midi(self, mock_port):
        return MidiOut(port=mock_port)

    def test_note_on(self, midi, mock_port):
        midi.note_on(channel=0, note=60, velocity=100)
        msg = mock_port.send.call_args[0][0]
        assert msg.type == "note_on"
        assert msg.channel == 0
        assert msg.note == 60
        assert msg.velocity == 100

    def test_note_off(self, midi, mock_port):
        midi.note_off(channel=0, note=60)
        msg = mock_port.send.call_args[0][0]
        assert msg.type == "note_off"
        assert msg.channel == 0
        assert msg.note == 60

    def test_control_change(self, midi, mock_port):
        midi.control_change(channel=0, control=74, value=100)
        msg = mock_port.send.call_args[0][0]
        assert msg.type == "control_change"
        assert msg.channel == 0
        assert msg.control == 74
        assert msg.value == 100

    def test_program_change(self, midi, mock_port):
        midi.program_change(channel=0, program=11)
        msg = mock_port.send.call_args[0][0]
        assert msg.type == "program_change"
        assert msg.channel == 0
        assert msg.program == 11

    def test_all_notes_off(self, midi, mock_port):
        midi.all_notes_off(channel=0)
        msg = mock_port.send.call_args[0][0]
        assert msg.type == "control_change"
        assert msg.control == 123  # All Notes Off CC
        assert msg.value == 0

    def test_velocity_clamped_to_127(self, midi, mock_port):
        midi.note_on(channel=0, note=60, velocity=200)
        msg = mock_port.send.call_args[0][0]
        assert msg.velocity == 127

    def test_velocity_clamped_to_0(self, midi, mock_port):
        midi.note_on(channel=0, note=60, velocity=-5)
        msg = mock_port.send.call_args[0][0]
        assert msg.velocity == 0
