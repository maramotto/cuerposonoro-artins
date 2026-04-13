"""cuerposonoro-jetson: body movement → real-time sound.

Entry point. Reads config, initialises all components, runs the main loop.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
from typing import Protocol

import yaml

from audio.chords import ChordProgression
from audio.gesture_mode import GestureMidiMode
from audio.jetson_sender import JetsonMidiSender
from audio.musical_mode import MusicalMidiMode
from audio.platform import make_fluidsynth_manager
from audio.realtime_mode import RealtimeMidiMode
from vision.capture import WebcamCamera
from vision.detector import PoseDetector
from vision.landmarks import Landmarks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("cuerposonoro")


class MidiMode(Protocol):
    """Common interface for all MIDI modes."""

    def update(self, landmarks: Landmarks) -> None: ...
    def close(self) -> None: ...


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


def _build_mode(midi_mode: str, config: dict, fluidsynth) -> MidiMode:
    """Construct the appropriate MIDI mode from config."""
    if midi_mode == "gesture":
        return GestureMidiMode(synth=fluidsynth.synth, config=config)

    if midi_mode == "realtime":
        return RealtimeMidiMode(synth=fluidsynth.synth, config=config)

    progression = ChordProgression.from_config(config["harmony"]["chord_progression"])

    if midi_mode == "jetson":
        return JetsonMidiSender(
            synth=fluidsynth.synth,
            config=config,
            chord_progression=progression,
        )

    # Default: musical
    return MusicalMidiMode(
        synth=fluidsynth.synth,
        config=config,
        chord_progression=progression,
    )


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

    # Build mode
    mode = _build_mode(midi_mode, config, fluidsynth)
    log.info("MIDI mode active: %s", midi_mode)

    # Camera
    camera = WebcamCamera(config["camera"]["device_id"])
    if not camera.is_opened:
        log.error("Cannot open camera")
        sys.exit(1)
    log.info("Camera opened")

    # Run the shared detection → mode loop
    _run_loop(camera, detector, mode, fluidsynth)


def _run_loop(camera: WebcamCamera, detector: PoseDetector, mode: MidiMode, fluidsynth) -> None:
    """Shared main loop: camera → detection → person tracking → mode update."""
    person_landmarks: dict[int, Landmarks] = {}
    running = True

    def on_signal(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    max_consecutive_failures = 30
    consecutive_failures = 0

    log.info("Main loop started")

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
             "or 'musical' (per-frame, chord progression)",
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
