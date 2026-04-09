from audio.fluidsynth import FluidsynthManager
from audio.platform import make_fluidsynth_manager


class TestMakeFluidsynthManager:
    def test_returns_fluidsynth_manager(self):
        mgr = make_fluidsynth_manager(
            soundfont="test.sf2", gain=0.5, sample_rate=44100,
        )
        assert isinstance(mgr, FluidsynthManager)

    def test_passes_driver_parameter(self):
        mgr = make_fluidsynth_manager(
            soundfont="test.sf2", gain=0.5, sample_rate=44100, driver="coreaudio",
        )
        assert mgr._driver == "coreaudio"

    def test_default_driver_is_pulseaudio(self):
        mgr = make_fluidsynth_manager(
            soundfont="test.sf2", gain=0.5, sample_rate=44100,
        )
        assert mgr._driver == "pulseaudio"
