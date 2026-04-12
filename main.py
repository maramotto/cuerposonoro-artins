"""cuerposonoro-jetson: body movement → real-time sound.

Entry point. Reads config, initialises all components, runs the main loop.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

import numpy as np
import yaml

from audio.chords import ChordProgression
from audio.jetson_sender import JetsonMidiSender
from audio.gesture_mode import GestureMidiMode
from audio.realtime_mode import RealtimeMidiMode
from audio.platform import make_fluidsynth_manager
from audio.midi import MidiOut
from features.arms import ArmFeatures
from features.harmony import HarmonyFeatures
from features.legs import LegFeatures
from features.silence import SilenceTracker
from vision.detector import PoseDetector
from vision.capture import WebcamCamera
from vision.landmarks import Landmarks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("cuerposonoro")


def load_config(path: str = "config.yaml") -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        log.critical("Config file not found: %s", path)
        sys.exit(1)
    except yaml.YAMLError as exc:
        log.critical("Invalid YAML in config file %s: %s", path, exc)
        sys.exit(1)


def run(config: dict, midi_mode: str = "musical") -> None:
    # Vision
    detector = PoseDetector(
        model_path=config["vision"]["model"],
        confidence=config["vision"]["confidence_threshold"],
        use_tensorrt=config["vision"].get("use_tensorrt", True),
        tensorrt_half=config["vision"].get("tensorrt_half", True),
    )

    # Audio
    fluidsynth = make_fluidsynth_manager(
        soundfont=config["fluidsynth"]["soundfont"],
        gain=config["fluidsynth"]["gain"],
        sample_rate=config["fluidsynth"]["sample_rate"],
        driver=config["fluidsynth"]["driver"],
    )
    fluidsynth.start()
    log.info("Fluidsynth started")

    midi = MidiOut(fluidsynth.synth)
    log.info("MIDI connected to Fluidsynth synth (direct API)")

    # Set programs
    mel_cfg = config["melody"]
    bass_cfg = config["bass"]
    midi.program_change(channel=mel_cfg["channel"], program=mel_cfg["program"])
    midi.program_change(channel=bass_cfg["channel"], program=bass_cfg["program"])

    # Harmony
    progression = ChordProgression.from_config(config["harmony"]["chord_progression"])

    if midi_mode == "gesture":
        _run_gesture(config, detector, fluidsynth)
    elif midi_mode == "realtime":
        _run_realtime(config, detector, fluidsynth)
    elif midi_mode == "jetson":
        _run_jetson(config, detector, fluidsynth, progression)
    else:
        _run_musical(config, detector, fluidsynth, progression)


def _run_gesture(config, detector, fluidsynth) -> None:
    """Gesture mode: direction-based, 3 voices, Re dorian."""
    mode = GestureMidiMode(synth=fluidsynth.synth, config=config)
    log.info(
        "Gesture MIDI mode active (hysteresis=%d, silence=%d frames)",
        config["gesture"]["hysteresis_frames"],
        config["gesture"]["silence_frames"],
    )

    camera = WebcamCamera(config["camera"]["device_id"])
    if not camera.is_opened:
        log.error("Cannot open camera")
        sys.exit(1)
    log.info("Camera opened")

    person_landmarks: dict[int, Landmarks] = {}
    running = True

    def on_signal(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    max_consecutive_failures = 30
    consecutive_failures = 0

    log.info("Main loop started (gesture mode, Re dorian)")

    try:
        while running:
            frame = camera.read()
            if frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    log.error(
                        "Camera feed lost after %d consecutive failures, exiting",
                        consecutive_failures,
                    )
                    break
                continue
            consecutive_failures = 0

            detections = detector.detect(frame)
            seen = set()

            for i, keypoints in enumerate(detections):
                seen.add(i)
                if i in person_landmarks:
                    person_landmarks[i].update(keypoints)
                else:
                    person_landmarks[i] = Landmarks(keypoints)

                mode.update(person_landmarks[i])

            disappeared = set(person_landmarks.keys()) - seen
            for i in disappeared:
                del person_landmarks[i]

    finally:
        mode.close()
        camera.release()
        fluidsynth.stop()
        log.info("Shutdown complete")


def _run_realtime(config, detector, fluidsynth) -> None:
    """Realtime mode: per-keypoint velocity-driven, D minor pentatonic."""
    mode = RealtimeMidiMode(synth=fluidsynth.synth, config=config)
    log.info(
        "Realtime MIDI mode active (hysteresis=%d, silence=%d frames)",
        config["realtime"]["hysteresis_frames"],
        config["realtime"]["silence_frames"],
    )

    camera = WebcamCamera(config["camera"]["device_id"])
    if not camera.is_opened:
        log.error("Cannot open camera")
        sys.exit(1)
    log.info("Camera opened")

    person_landmarks: dict[int, Landmarks] = {}
    running = True

    def on_signal(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    max_consecutive_failures = 30
    consecutive_failures = 0

    log.info("Main loop started (realtime mode)")

    try:
        while running:
            frame = camera.read()
            if frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    log.error(
                        "Camera feed lost after %d consecutive failures, exiting",
                        consecutive_failures,
                    )
                    break
                continue
            consecutive_failures = 0

            detections = detector.detect(frame)
            seen = set()

            for i, keypoints in enumerate(detections):
                seen.add(i)
                if i in person_landmarks:
                    person_landmarks[i].update(keypoints)
                else:
                    person_landmarks[i] = Landmarks(keypoints)

                mode.update(person_landmarks[i])

            disappeared = set(person_landmarks.keys()) - seen
            for i in disappeared:
                del person_landmarks[i]

    finally:
        mode.close()
        camera.release()
        fluidsynth.stop()
        log.info("Shutdown complete")


def _run_jetson(config, detector, fluidsynth, progression) -> None:
    """Jetson mode: velocity-driven, sustained notes, hysteresis."""
    sender = JetsonMidiSender(
        synth=fluidsynth.synth,
        config=config,
        chord_progression=progression,
    )
    log.info("Jetson MIDI mode active (hysteresis=%d frames)",
             config["jetson"]["hysteresis_frames"])

    camera = WebcamCamera(config["camera"]["device_id"])
    if not camera.is_opened:
        log.error("Cannot open camera")
        sys.exit(1)
    log.info("Camera opened")

    person_landmarks: dict[int, Landmarks] = {}
    running = True

    def on_signal(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    max_consecutive_failures = 30
    consecutive_failures = 0

    log.info("Main loop started — chord: %s", progression.current.name)

    try:
        while running:
            frame = camera.read()
            if frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    log.error(
                        "Camera feed lost after %d consecutive failures, exiting",
                        consecutive_failures,
                    )
                    break
                continue
            consecutive_failures = 0

            detections = detector.detect(frame)
            seen = set()

            for i, keypoints in enumerate(detections):
                seen.add(i)
                if i in person_landmarks:
                    person_landmarks[i].update(keypoints)
                else:
                    person_landmarks[i] = Landmarks(keypoints)

                sender.update(person_landmarks[i])

            # Clean up disappeared people
            disappeared = set(person_landmarks.keys()) - seen
            for i in disappeared:
                del person_landmarks[i]

    finally:
        sender.close()
        camera.release()
        fluidsynth.stop()
        log.info("Shutdown complete")


def _run_musical(config, detector, fluidsynth, progression) -> None:
    """Musical mode: per-frame note triggers (original behaviour)."""
    midi = MidiOut(fluidsynth.synth)
    log.info("MIDI connected to Fluidsynth synth (direct API)")

    mel_cfg = config["melody"]
    bass_cfg = config["bass"]
    harm_cfg = config["harmony"]
    midi.program_change(channel=mel_cfg["channel"], program=mel_cfg["program"])
    midi.program_change(channel=bass_cfg["channel"], program=bass_cfg["program"])

    silence_tracker = SilenceTracker(
        threshold=config["silence"]["velocity_threshold"],
        timeout_ms=config["silence"]["timeout_ms"],
    )

    camera = WebcamCamera(config["camera"]["device_id"])
    if not camera.is_opened:
        log.error("Cannot open camera")
        sys.exit(1)
    log.info("Camera opened")

    person_landmarks: dict[int, Landmarks] = {}
    active_melody_notes: dict[int, int] = {}
    active_bass_notes: dict[int, int] = {}

    running = True

    def on_signal(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    max_consecutive_failures = 30
    consecutive_failures = 0

    log.info("Main loop started — chord: %s", progression.current.name)

    try:
        while running:
            frame = camera.read()
            if frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    log.error(
                        "Camera feed lost after %d consecutive failures, exiting",
                        consecutive_failures,
                    )
                    break
                continue
            consecutive_failures = 0

            detections = detector.detect(frame)
            seen = set()

            for i, keypoints in enumerate(detections):
                seen.add(i)
                if i in person_landmarks:
                    person_landmarks[i].update(keypoints)
                else:
                    person_landmarks[i] = Landmarks(keypoints)

                lm = person_landmarks[i]

                body_vel = lm.mean_velocity(list(range(17)))
                is_silent = silence_tracker.update(body_vel)

                if is_silent:
                    if i in active_melody_notes:
                        midi.note_off(mel_cfg["channel"], active_melody_notes.pop(i))
                    if i in active_bass_notes:
                        midi.note_off(bass_cfg["channel"], active_bass_notes.pop(i))
                    continue

                chord = progression.current

                harmony = HarmonyFeatures(lm)
                if harmony.should_advance(harm_cfg["torso_tilt_threshold"]):
                    progression.advance()
                    log.info("Chord → %s", progression.current.name)
                elif harmony.should_retreat(harm_cfg["torso_tilt_threshold"]):
                    progression.retreat()
                    log.info("Chord → %s", progression.current.name)

                head_tilt = harmony.head_tilt()
                tilt_sign = 0.0
                if head_tilt > harm_cfg["head_tilt_threshold"]:
                    tilt_sign = 1.0
                elif head_tilt < -harm_cfg["head_tilt_threshold"]:
                    tilt_sign = -1.0

                arms = ArmFeatures(lm)
                arm_vel = arms.arm_velocity()

                if arm_vel > mel_cfg["trigger_threshold"]:
                    if i in active_melody_notes:
                        midi.note_off(mel_cfg["channel"], active_melody_notes[i])

                    note = chord.note_from_height(
                        arms.mean_wrist_height(),
                        mel_cfg["note_min"],
                        mel_cfg["note_max"],
                        tilt=tilt_sign,
                    )
                    velocity = int(np.interp(
                        arm_vel,
                        [mel_cfg["trigger_threshold"], 0.3],
                        [mel_cfg["velocity_min"], mel_cfg["velocity_max"]],
                    ))
                    velocity = max(0, min(127, velocity))
                    midi.note_on(mel_cfg["channel"], note, velocity)
                    active_melody_notes[i] = note

                midi.control_change(
                    mel_cfg["channel"], mel_cfg["brightness_cc"], arms.brightness(),
                )

                legs = LegFeatures(lm)
                ankle_vel = legs.ankle_velocity()

                if ankle_vel > bass_cfg["trigger_threshold"]:
                    if i in active_bass_notes:
                        midi.note_off(bass_cfg["channel"], active_bass_notes[i])

                    bass_note = chord.root
                    midi.note_on(bass_cfg["channel"], bass_note, bass_cfg["velocity"])
                    active_bass_notes[i] = bass_note

            disappeared = set(person_landmarks.keys()) - seen
            for i in disappeared:
                if i in active_melody_notes:
                    midi.note_off(mel_cfg["channel"], active_melody_notes.pop(i))
                if i in active_bass_notes:
                    midi.note_off(bass_cfg["channel"], active_bass_notes.pop(i))
                del person_landmarks[i]

    finally:
        for ch in [mel_cfg["channel"], bass_cfg["channel"]]:
            midi.all_notes_off(ch)
        camera.release()
        fluidsynth.stop()
        log.info("Shutdown complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CuerpoSonoro: body movement to real-time sound",
    )
    parser.add_argument(
        "--mode",
        choices=["midi"],
        default="midi",
        help="Audio engine mode (default: midi)",
    )
    parser.add_argument(
        "--midi-mode",
        choices=["gesture", "realtime", "jetson", "musical"],
        default="gesture",
        help="MIDI mapping strategy: 'gesture' (direction-based, Re dorian), "
             "'realtime' (per-keypoint velocity), 'jetson' (velocity-driven), "
             "or 'musical' (tempo-grid)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log.info("Starting with mode=%s, midi_mode=%s", args.mode, args.midi_mode)
    config = load_config(args.config)
    run(config, midi_mode=args.midi_mode)


if __name__ == "__main__":
    main()
