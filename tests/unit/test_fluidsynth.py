import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from audio.fluidsynth import FluidsynthManager


class TestFluidsynthManager:
    def _make_manager(self, **overrides) -> FluidsynthManager:
        defaults = {
            "soundfont": "JJazzLab-SoundFont.sf2",
            "gain": 0.8,
            "sample_rate": 44100,
            "driver": "pulseaudio",
        }
        defaults.update(overrides)
        return FluidsynthManager(**defaults)

    @patch("audio.fluidsynth.fluidsynth")
    def test_start_initialises_synth_and_loads_soundfont(self, mock_fs_module):
        mock_synth = MagicMock()
        mock_fs_module.Synth.return_value = mock_synth
        mock_synth.sfload.return_value = 1

        mgr = self._make_manager()
        mgr.start()

        mock_fs_module.Synth.assert_called_once_with(gain=0.8, samplerate=44100)
        mock_synth.start.assert_called_once_with(driver="pulseaudio")
        mock_synth.sfload.assert_called_once_with("JJazzLab-SoundFont.sf2")

    @patch("audio.fluidsynth.fluidsynth")
    def test_stop_deletes_synth_and_sets_running_false(self, mock_fs_module):
        mock_synth = MagicMock()
        mock_fs_module.Synth.return_value = mock_synth
        mock_synth.sfload.return_value = 1

        mgr = self._make_manager()
        mgr.start()
        assert mgr.running
        mgr.stop()
        mock_synth.delete.assert_called_once()
        assert not mgr.running

    @patch("audio.fluidsynth.fluidsynth")
    def test_running_false_before_start(self, mock_fs_module):
        mgr = self._make_manager()
        assert not mgr.running

    @patch("audio.fluidsynth.fluidsynth")
    def test_running_true_after_start(self, mock_fs_module):
        mock_synth = MagicMock()
        mock_fs_module.Synth.return_value = mock_synth
        mock_synth.sfload.return_value = 1

        mgr = self._make_manager()
        mgr.start()
        assert mgr.running

    @patch("audio.fluidsynth.fluidsynth")
    def test_running_false_after_stop(self, mock_fs_module):
        mock_synth = MagicMock()
        mock_fs_module.Synth.return_value = mock_synth
        mock_synth.sfload.return_value = 1

        mgr = self._make_manager()
        mgr.start()
        mgr.stop()
        assert not mgr.running

    @patch("audio.fluidsynth.fluidsynth")
    def test_synth_property_returns_synth_after_start(self, mock_fs_module):
        mock_synth = MagicMock()
        mock_fs_module.Synth.return_value = mock_synth
        mock_synth.sfload.return_value = 1

        mgr = self._make_manager()
        mgr.start()
        assert mgr.synth is mock_synth

    @patch("audio.fluidsynth.fluidsynth")
    def test_synth_property_is_none_before_start(self, mock_fs_module):
        mgr = self._make_manager()
        assert mgr.synth is None

    @patch("audio.fluidsynth.fluidsynth")
    def test_start_raises_if_soundfont_not_found(self, mock_fs_module):
        mock_synth = MagicMock()
        mock_fs_module.Synth.return_value = mock_synth
        mock_synth.sfload.return_value = -1  # fluidsynth returns -1 on failure

        mgr = self._make_manager(soundfont="nonexistent.sf2")
        with pytest.raises(RuntimeError, match="soundfont"):
            mgr.start()

    @patch("audio.fluidsynth.fluidsynth")
    def test_stop_without_start_is_safe(self, mock_fs_module):
        mgr = self._make_manager()
        mgr.stop()  # Should not raise

    @patch("audio.fluidsynth.fluidsynth")
    def test_driver_passed_to_start(self, mock_fs_module):
        mock_synth = MagicMock()
        mock_fs_module.Synth.return_value = mock_synth
        mock_synth.sfload.return_value = 1

        mgr = self._make_manager(driver="coreaudio")
        mgr.start()
        mock_synth.start.assert_called_once_with(driver="coreaudio")

    @patch("audio.fluidsynth.fluidsynth")
    def test_gain_passed_to_synth(self, mock_fs_module):
        mock_synth = MagicMock()
        mock_fs_module.Synth.return_value = mock_synth
        mock_synth.sfload.return_value = 1

        mgr = self._make_manager(gain=0.5)
        mgr.start()
        mock_fs_module.Synth.assert_called_once_with(gain=0.5, samplerate=44100)
