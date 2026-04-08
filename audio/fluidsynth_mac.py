from __future__ import annotations

import logging
import subprocess

log = logging.getLogger(__name__)


class FluidsynthManagerMac:
    """Manages a headless Fluidsynth process on macOS (CoreAudio + CoreMIDI)."""

    def __init__(self, soundfont: str, gain: float, sample_rate: int) -> None:
        self._soundfont = soundfont
        self._gain = gain
        self._sample_rate = sample_rate
        self._process: subprocess.Popen | None = None

    def start(self) -> None:
        self._process = subprocess.Popen(
            [
                "fluidsynth",
                "-a", "coreaudio",
                "-m", "coremidi",
                "-g", str(self._gain),
                "-r", str(self._sample_rate),
                "-ni",
                self._soundfont,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log.warning("Fluidsynth did not exit cleanly, killing")
            self._process.kill()
            self._process.wait()
        finally:
            self._process = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def connect_midi_port(self, virtual_port_name: str) -> None:
        """Log manual MIDI connection instructions for macOS.

        macOS has no ``aconnect``; the user must wire MIDI in Audio MIDI Setup.
        """
        log.info(
            "Mac mode: open Audio MIDI Setup → MIDI Studio and connect "
            "'%s' to the Fluidsynth input.",
            virtual_port_name,
        )
