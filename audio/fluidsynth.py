from __future__ import annotations

import logging

import fluidsynth

log = logging.getLogger(__name__)


class FluidsynthManager:
    """Manages an in-process Fluidsynth synth via pyfluidsynth."""

    def __init__(
        self, soundfont: str, gain: float, sample_rate: int, driver: str = "pulseaudio"
    ) -> None:
        self._soundfont = soundfont
        self._gain = gain
        self._sample_rate = sample_rate
        self._driver = driver
        self._synth: fluidsynth.Synth | None = None

    def start(self) -> None:
        self._synth = fluidsynth.Synth(gain=self._gain, samplerate=self._sample_rate)
        self._synth.start(driver=self._driver)
        sfid = self._synth.sfload(self._soundfont)
        if sfid == -1:
            self._synth.delete()
            self._synth = None
            raise RuntimeError(
                f"Failed to load soundfont: {self._soundfont}"
            )
        log.info("Fluidsynth started (driver=%s, soundfont=%s)", self._driver, self._soundfont)

    def stop(self) -> None:
        if self._synth is None:
            return
        self._synth.delete()
        self._synth = None
        log.info("Fluidsynth stopped")

    @property
    def running(self) -> bool:
        return self._synth is not None

    @property
    def synth(self) -> fluidsynth.Synth | None:
        return self._synth
