import pytest
from unittest.mock import MagicMock, patch
from audio.fluidsynth import FluidsynthManager


class TestFluidsynthManager:
    @patch("audio.fluidsynth.subprocess")
    def test_start_launches_process(self, mock_subprocess):
        mock_subprocess.Popen.return_value = MagicMock()
        mgr = FluidsynthManager(
            soundfont="JJazzLab-SoundFont.sf2",
            gain=0.8,
            sample_rate=44100,
        )
        mgr.start()
        mock_subprocess.Popen.assert_called_once()
        args = mock_subprocess.Popen.call_args[0][0]
        assert "fluidsynth" in args[0]
        assert "JJazzLab-SoundFont.sf2" in args

    @patch("audio.fluidsynth.subprocess")
    def test_stop_terminates_process(self, mock_subprocess):
        mock_proc = MagicMock()
        mock_subprocess.Popen.return_value = mock_proc
        mgr = FluidsynthManager(
            soundfont="JJazzLab-SoundFont.sf2",
            gain=0.8,
            sample_rate=44100,
        )
        mgr.start()
        mgr.stop()
        mock_proc.terminate.assert_called_once()

    @patch("audio.fluidsynth.subprocess")
    def test_stop_without_start_is_safe(self, mock_subprocess):
        mgr = FluidsynthManager(
            soundfont="JJazzLab-SoundFont.sf2",
            gain=0.8,
            sample_rate=44100,
        )
        mgr.stop()  # Should not raise

    @patch("audio.fluidsynth.subprocess")
    def test_gain_in_args(self, mock_subprocess):
        mock_subprocess.Popen.return_value = MagicMock()
        mgr = FluidsynthManager(
            soundfont="JJazzLab-SoundFont.sf2",
            gain=0.5,
            sample_rate=44100,
        )
        mgr.start()
        args = mock_subprocess.Popen.call_args[0][0]
        gain_idx = args.index("-g")
        assert args[gain_idx + 1] == "0.5"
