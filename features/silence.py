from __future__ import annotations

import time


class SilenceTracker:
    """Track whether body is still enough to trigger silence.

    When body velocity stays below ``threshold`` for longer than
    ``timeout_ms`` milliseconds, the tracker reports silence.
    Any velocity above the threshold resets the timer.
    """

    def __init__(self, threshold: float, timeout_ms: int) -> None:
        self._threshold = threshold
        self._timeout_s = timeout_ms / 1000.0
        self._still_since: float | None = None

    def update(self, velocity: float) -> bool:
        """Return True if sound should be silent."""
        if velocity < self._threshold:
            if self._still_since is None:
                self._still_since = time.monotonic()
            return (time.monotonic() - self._still_since) >= self._timeout_s
        else:
            self._still_since = None
            return False
