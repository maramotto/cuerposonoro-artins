from unittest.mock import patch

from audio.fluidsynth import FluidsynthManager
from audio.fluidsynth_mac import FluidsynthManagerMac
from audio.platform import make_fluidsynth_manager


class TestMakeFluidsynthManager:
    def test_returns_mac_manager_on_darwin(self):
        with patch("audio.platform.sys") as mock_sys:
            mock_sys.platform = "darwin"
            mgr = make_fluidsynth_manager(
                soundfont="test.sf2", gain=0.5, sample_rate=44100,
            )
        assert isinstance(mgr, FluidsynthManagerMac)

    def test_returns_jetson_manager_on_linux(self):
        with patch("audio.platform.sys") as mock_sys:
            mock_sys.platform = "linux"
            mgr = make_fluidsynth_manager(
                soundfont="test.sf2", gain=0.5, sample_rate=44100,
            )
        assert isinstance(mgr, FluidsynthManager)
