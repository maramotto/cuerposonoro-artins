import sys
import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# Mock ultralytics before any import that touches it
sys.modules["ultralytics"] = MagicMock()

from audio.chords import Chord, ChordProgression
from vision.landmarks import Landmarks
from features.arms import ArmFeatures
from features.legs import LegFeatures
from features.harmony import HarmonyFeatures


class TestSilenceDetection:
    """Test the silence logic: no movement for > timeout → silence."""

    def test_body_below_threshold_triggers_silence(self):
        """If all body velocity is below threshold for > timeout, silence."""
        kp1 = np.zeros((17, 3), dtype=np.float32)
        kp1[:, 2] = 1.0
        kp2 = kp1.copy()
        # Tiny movement, below any reasonable threshold
        kp2[9, 0] += 0.001

        lm = Landmarks(kp1)
        lm.update(kp2)

        total_velocity = lm.mean_velocity(list(range(17)))
        threshold = 0.02
        assert total_velocity < threshold

    def test_body_above_threshold_produces_sound(self):
        kp1 = np.zeros((17, 3), dtype=np.float32)
        kp1[:, 2] = 1.0
        kp2 = kp1.copy()
        # Significant arm movement
        kp2[9, 0] += 0.3
        kp2[10, 0] -= 0.3

        lm = Landmarks(kp1)
        lm.update(kp2)

        total_velocity = lm.mean_velocity(list(range(17)))
        threshold = 0.02
        assert total_velocity > threshold


class TestEndToEndMapping:
    """Test the full pipeline: landmarks → features → MIDI values."""

    def _make_person(self, wrist_y=0.3, ankle_move=0.2):
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[:, 2] = 1.0
        kp[3] = [0.4, 0.1, 1.0]   # left ear
        kp[4] = [0.6, 0.1, 1.0]   # right ear
        kp[5] = [0.3, 0.3, 1.0]   # left shoulder
        kp[6] = [0.7, 0.3, 1.0]   # right shoulder
        kp[7] = [0.2, 0.4, 1.0]   # left elbow
        kp[8] = [0.8, 0.4, 1.0]   # right elbow
        kp[9] = [0.1, wrist_y, 1.0]   # left wrist
        kp[10] = [0.9, wrist_y, 1.0]  # right wrist
        kp[11] = [0.4, 0.6, 1.0]  # left hip
        kp[12] = [0.6, 0.6, 1.0]  # right hip
        kp[13] = [0.4, 0.7, 1.0]  # left knee
        kp[14] = [0.6, 0.7, 1.0]  # right knee
        kp[15] = [0.4, 0.9, 1.0]  # left ankle
        kp[16] = [0.6, 0.9, 1.0]  # right ankle
        return kp

    def test_arms_up_produces_high_note(self):
        chord = Chord("Dm9", 62, [62, 65, 69, 72, 76])
        kp = self._make_person(wrist_y=0.1)  # arms up
        lm = Landmarks(kp)
        arms = ArmFeatures(lm)

        note = chord.note_from_height(arms.mean_wrist_height(), 48, 84)
        # Arms up → high height → high note
        assert note >= 72

    def test_arms_down_produces_low_note(self):
        chord = Chord("Dm9", 62, [62, 65, 69, 72, 76])
        kp = self._make_person(wrist_y=0.9)  # arms down
        lm = Landmarks(kp)
        arms = ArmFeatures(lm)

        note = chord.note_from_height(arms.mean_wrist_height(), 48, 84)
        # Arms down → low height → low note
        assert note <= 55

    def test_walking_triggers_bass(self):
        kp1 = self._make_person()
        kp2 = self._make_person()
        kp2[15, 0] += 0.2  # left ankle moves
        kp2[16, 0] += 0.2  # right ankle moves

        lm = Landmarks(kp1)
        lm.update(kp2)
        legs = LegFeatures(lm)

        assert legs.ankle_velocity() > 0.03  # above trigger threshold

    def test_progression_advances_on_torso_tilt(self):
        prog = ChordProgression.from_config([
            {"name": "Dm9", "root": 62, "notes": [62, 65, 69, 72, 76]},
            {"name": "G13sus4", "root": 67, "notes": [67, 72, 74, 76, 81]},
        ])
        kp = self._make_person()
        # Shift shoulders right
        kp[5, 0] = 0.5
        kp[6, 0] = 0.9
        lm = Landmarks(kp)
        harmony = HarmonyFeatures(lm)

        if harmony.should_advance(threshold=0.05):
            prog.advance()

        assert prog.current.name == "G13sus4"
