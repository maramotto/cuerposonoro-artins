import logging
import subprocess as real_subprocess
from unittest.mock import MagicMock, patch

import pytest

from audio.fluidsynth_mac import FluidsynthManagerMac


class TestFluidsynthManagerMac:
    def _make_manager(self) -> FluidsynthManagerMac:
        return FluidsynthManagerMac(
            soundfont="JJazzLab-SoundFont.sf2",
            gain=0.8,
            sample_rate=44100,
        )

    @patch("audio.fluidsynth_mac.subprocess")
    def test_start_uses_coreaudio_and_coremidi(self, mock_subprocess):
        """start() launches Fluidsynth with -a coreaudio and -m coremidi."""
        mock_subprocess.Popen.return_value = MagicMock()
        mgr = self._make_manager()
        mgr.start()
        mock_subprocess.Popen.assert_called_once()
        args = mock_subprocess.Popen.call_args[0][0]
        assert "-a" in args
        assert args[args.index("-a") + 1] == "coreaudio"
        assert "-m" in args
        assert args[args.index("-m") + 1] == "coremidi"

    @patch("audio.fluidsynth_mac.subprocess")
    def test_start_does_not_use_alsa(self, mock_subprocess):
        """start() must NOT include -a alsa or -m alsa_seq."""
        mock_subprocess.Popen.return_value = MagicMock()
        mgr = self._make_manager()
        mgr.start()
        args = mock_subprocess.Popen.call_args[0][0]
        assert "alsa" not in args
        assert "alsa_seq" not in args

    @patch("audio.fluidsynth_mac.subprocess")
    def test_stop_terminates_process(self, mock_subprocess):
        """stop() terminates the Fluidsynth process."""
        mock_proc = MagicMock()
        mock_subprocess.Popen.return_value = mock_proc
        mgr = self._make_manager()
        mgr.start()
        mgr.stop()
        mock_proc.terminate.assert_called_once()

    @patch("audio.fluidsynth_mac.subprocess")
    def test_stop_kills_on_timeout(self, mock_subprocess):
        """If Fluidsynth hangs on terminate, kill it and clear state."""
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = [
            real_subprocess.TimeoutExpired("fluidsynth", 5),
            None,
        ]
        mock_subprocess.Popen.return_value = mock_proc
        mock_subprocess.TimeoutExpired = real_subprocess.TimeoutExpired
        mgr = self._make_manager()
        mgr.start()
        mgr.stop()
        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
        assert not mgr.running

    @patch("audio.fluidsynth_mac.subprocess")
    def test_connect_midi_port_does_not_call_subprocess_run(self, mock_subprocess):
        """connect_midi_port() must NOT call subprocess.run (no aconnect on Mac)."""
        mock_subprocess.Popen.return_value = MagicMock()
        mgr = self._make_manager()
        mgr.start()
        mgr.connect_midi_port("cuerposonoro")
        mock_subprocess.run.assert_not_called()

    @patch("audio.fluidsynth_mac.subprocess")
    def test_connect_midi_port_logs_audio_midi_setup(self, mock_subprocess, caplog):
        """connect_midi_port() logs a message containing 'Audio MIDI Setup'."""
        mock_subprocess.Popen.return_value = MagicMock()
        mgr = self._make_manager()
        mgr.start()
        with caplog.at_level(logging.INFO):
            mgr.connect_midi_port("cuerposonoro")
        assert any("Audio MIDI Setup" in record.message for record in caplog.records)

    @patch("audio.fluidsynth_mac.subprocess")
    def test_running_property(self, mock_subprocess):
        """running returns False before start and True after start."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # process alive
        mock_subprocess.Popen.return_value = mock_proc
        mgr = self._make_manager()
        assert not mgr.running
        mgr.start()
        assert mgr.running
