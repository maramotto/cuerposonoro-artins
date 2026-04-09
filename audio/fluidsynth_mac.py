from __future__ import annotations

from audio.fluidsynth import FluidsynthManager


# FluidsynthManagerMac is now an alias for FluidsynthManager.
# The audio driver (coreaudio vs pulseaudio) is configured via config.yaml,
# so no platform-specific class is needed.
FluidsynthManagerMac = FluidsynthManager
