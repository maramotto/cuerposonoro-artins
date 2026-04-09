import pytest
from unittest.mock import MagicMock

from audio.midi import MidiOut


class TestMidiOut:
    @pytest.fixture
    def mock_synth(self):
        return MagicMock()

    @pytest.fixture
    def midi(self, mock_synth):
        return MidiOut(synth=mock_synth)

    def test_note_on(self, midi, mock_synth):
        midi.note_on(channel=0, note=60, velocity=100)
        mock_synth.noteon.assert_called_once_with(0, 60, 100)

    def test_note_off(self, midi, mock_synth):
        midi.note_off(channel=0, note=60)
        mock_synth.noteoff.assert_called_once_with(0, 60)

    def test_control_change(self, midi, mock_synth):
        midi.control_change(channel=0, control=74, value=100)
        mock_synth.cc.assert_called_once_with(0, 74, 100)

    def test_program_change(self, midi, mock_synth):
        midi.program_change(channel=0, program=11)
        mock_synth.program_change.assert_called_once_with(0, 11)

    def test_all_notes_off(self, midi, mock_synth):
        midi.all_notes_off(channel=0)
        mock_synth.cc.assert_called_once_with(0, 123, 0)

    def test_velocity_clamped_to_127(self, midi, mock_synth):
        midi.note_on(channel=0, note=60, velocity=200)
        mock_synth.noteon.assert_called_once_with(0, 60, 127)

    def test_velocity_clamped_to_0(self, midi, mock_synth):
        midi.note_on(channel=0, note=60, velocity=-5)
        mock_synth.noteon.assert_called_once_with(0, 60, 0)

    def test_note_on_different_channel(self, midi, mock_synth):
        midi.note_on(channel=1, note=48, velocity=80)
        mock_synth.noteon.assert_called_once_with(1, 48, 80)

    def test_control_change_brightness(self, midi, mock_synth):
        midi.control_change(channel=0, control=74, value=64)
        mock_synth.cc.assert_called_once_with(0, 74, 64)
