from __future__ import annotations

import logging
import re
import subprocess
import time

log = logging.getLogger(__name__)

# aconnect -l client line pattern: "client 128: 'FLUID Synth (31337)' [type=...]"
_CLIENT_RE = re.compile(r"^client\s+(\d+):\s+'([^']+)'")
# Port line pattern: "    0 'Synth input port (31337:0)'"
_PORT_RE = re.compile(r"^\s+(\d+)\s+'")

_MAX_RETRIES = 10
_INITIAL_BACKOFF_S = 0.5
_MAX_BACKOFF_S = 4.0


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
        """Connect a virtual MIDI port to the Fluidsynth ALSA input.

        Uses ``aconnect`` to discover ports and wire them together.
        Retries with exponential backoff if the Fluidsynth port is not
        yet visible (it takes a moment after process startup).
        """
        fluid_addr = None
        ports = ""
        backoff = _INITIAL_BACKOFF_S

        for attempt in range(_MAX_RETRIES):
            ports = self._list_alsa_ports()
            fluid_addr = self._find_port(ports, "FLUID Synth")
            if fluid_addr is not None:
                break
            log.info(
                "FLUID Synth port not found, retrying (%d/%d)...",
                attempt + 1,
                _MAX_RETRIES,
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, _MAX_BACKOFF_S)

        if fluid_addr is None:
            raise RuntimeError(
                f"FLUID Synth ALSA port not found after {_MAX_RETRIES} attempts"
            )

        # Use the same listing snapshot where fluid_addr was found
        virtual_addr = self._find_port(ports, virtual_port_name)
        if virtual_addr is None:
            raise RuntimeError(
                f"Virtual MIDI port '{virtual_port_name}' not found in ALSA"
            )

        result = subprocess.run(
            ["aconnect", virtual_addr, fluid_addr],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"aconnect failed ({result.returncode}): {result.stderr}"
            )

        log.info("MIDI connected: %s → %s", virtual_addr, fluid_addr)

    @staticmethod
    def _list_alsa_ports() -> str:
        result = subprocess.run(
            ["aconnect", "-l"],
            capture_output=True,
            text=True,
        )
        return result.stdout

    @staticmethod
    def _find_port(aconnect_output: str, name: str) -> str | None:
        """Find ``client:port`` address for a named client in aconnect output."""
        current_client: str | None = None
        for line in aconnect_output.splitlines():
            client_match = _CLIENT_RE.match(line)
            if client_match:
                client_id, client_name = client_match.groups()
                if name in client_name:
                    current_client = client_id
                else:
                    current_client = None
                continue
            if current_client is not None:
                port_match = _PORT_RE.match(line)
                if port_match:
                    return f"{current_client}:{port_match.group(1)}"
        return None
