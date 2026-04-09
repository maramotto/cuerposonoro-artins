from __future__ import annotations

from audio.fluidsynth import FluidsynthManager


def make_fluidsynth_manager(
    soundfont: str, gain: float, sample_rate: int, driver: str = "pulseaudio"
) -> FluidsynthManager:
    """Return a FluidsynthManager configured with the given audio driver.

    The driver parameter selects the audio backend:
      - "pulseaudio" for Jetson
      - "coreaudio" for Mac development
    """
    return FluidsynthManager(soundfont, gain, sample_rate, driver=driver)
