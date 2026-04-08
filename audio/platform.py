from __future__ import annotations

import sys

from audio.fluidsynth import FluidsynthManager
from audio.fluidsynth_mac import FluidsynthManagerMac


def make_fluidsynth_manager(
    soundfont: str, gain: float, sample_rate: int
) -> FluidsynthManager | FluidsynthManagerMac:
    """Return the platform-appropriate Fluidsynth manager."""
    if sys.platform == "darwin":
        return FluidsynthManagerMac(soundfont, gain, sample_rate)
    return FluidsynthManager(soundfont, gain, sample_rate)
