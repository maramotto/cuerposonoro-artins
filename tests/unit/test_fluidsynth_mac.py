from audio.fluidsynth import FluidsynthManager
from audio.fluidsynth_mac import FluidsynthManagerMac


class TestFluidsynthManagerMac:
    def test_mac_manager_is_same_class(self):
        """FluidsynthManagerMac is now an alias for FluidsynthManager."""
        assert FluidsynthManagerMac is FluidsynthManager
