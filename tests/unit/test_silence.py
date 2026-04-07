from unittest.mock import patch
from features.silence import SilenceTracker


class TestSilenceTracker:
    def _make_tracker(self, threshold: float = 0.02, timeout_ms: int = 500):
        return SilenceTracker(threshold=threshold, timeout_ms=timeout_ms)

    def test_above_threshold_is_not_silent(self):
        tracker = self._make_tracker()
        assert tracker.update(0.1) is False

    def test_below_threshold_not_immediately_silent(self):
        """Below threshold but before timeout — not yet silent."""
        tracker = self._make_tracker(timeout_ms=500)
        with patch("features.silence.time.monotonic", return_value=1000.0):
            assert tracker.update(0.01) is False

    def test_below_threshold_after_timeout_is_silent(self):
        """Below threshold for longer than timeout — silent."""
        tracker = self._make_tracker(timeout_ms=500)
        with patch("features.silence.time.monotonic", return_value=1000.0):
            tracker.update(0.01)  # starts the timer at t=1000
        with patch("features.silence.time.monotonic", return_value=1000.6):
            assert tracker.update(0.01) is True  # 600ms > 500ms

    def test_movement_resets_timer(self):
        """Going above threshold resets the stillness timer."""
        tracker = self._make_tracker(timeout_ms=500)
        with patch("features.silence.time.monotonic", return_value=1000.0):
            tracker.update(0.01)  # start timer
        with patch("features.silence.time.monotonic", return_value=1000.3):
            tracker.update(0.1)  # above threshold — reset
        with patch("features.silence.time.monotonic", return_value=1000.6):
            # Only 300ms since reset, should not be silent
            assert tracker.update(0.01) is False

    def test_exactly_at_threshold_is_not_silent(self):
        """Velocity exactly equal to threshold is not below it."""
        tracker = self._make_tracker(threshold=0.02, timeout_ms=0)
        assert tracker.update(0.02) is False

    def test_zero_timeout_is_immediate(self):
        """With timeout_ms=0, any below-threshold velocity is immediately silent."""
        tracker = self._make_tracker(timeout_ms=0)
        assert tracker.update(0.01) is True
