from __future__ import annotations

import subprocess


class FluidsynthManager:
    """Manages a headless Fluidsynth process."""

    def __init__(self, soundfont: str, gain: float, sample_rate: int) -> None:
        self._soundfont = soundfont
        self._gain = gain
        self._sample_rate = sample_rate
        self._process: subprocess.Popen | None = None

    def start(self) -> None:
        self._process = subprocess.Popen(
            [
                "fluidsynth",
                "-a", "alsa",
                "-m", "alsa_seq",
                "-g", str(self._gain),
                "-r", str(self._sample_rate),
                "-ni",
                self._soundfont,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> None:
        if self._process is not None:
            self._process.terminate()
            self._process.wait(timeout=5)
            self._process = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None
