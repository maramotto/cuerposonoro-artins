# cuerposonoro-jetson

An autonomous art installation that translates full-body human movement into
real-time sound. Built for the NVIDIA Jetson Orin Nano — plug in the power strip
and it runs. No screen, no keyboard, no manual intervention.

This is a **separate project** from
[cuerposonoro](https://github.com/mara/cuerposonoro) (the main thesis). It
adapts the body-to-sound mapping for a specific physical context with conscious
technical trade-offs. See `DESIGN.md` for the full artistic and technical
rationale.

## How it works

A USB camera watches the space. YOLOv8-Pose (accelerated via TensorRT on the
Jetson GPU) detects people and their 17 COCO body landmarks. Movement is
analysed in real time and mapped to MIDI, which Fluidsynth renders as audio
through a soundfont:

- **Arms** control a vibraphone melody (MIDI channel 1)
- **Legs** control an acoustic bass rhythm (MIDI channel 2)
- **Torso tilt** advances or retreats through a D minor chord progression
- **Head tilt** adds tension or simplifies the active chord
- **No movement = no sound.** Silence is part of the work.

Multiple people can interact simultaneously — YOLOv8-Pose is multi-person
by design.

### Signal flow

```
Camera → YOLOv8-Pose (TensorRT) → Landmark extraction
  → Feature analysis (arms, legs, harmony)
    → pyfluidsynth API → Fluidsynth synth (in-process)
      → PulseAudio → Bluetooth A2DP (ZK502-C) → Speakers
```

End-to-end latency target: **under 80ms** (camera frame to audible sound).

## Hardware

| Component | Specification |
|---|---|
| Compute | NVIDIA Jetson Orin Nano 8GB, JetPack 6.1, CUDA 12.6, TensorRT 10.3 |
| Camera | Logitech C922 (USB), appears as `/dev/video0` |
| Audio output | Bluetooth A2DP speaker (ZK502-C), auto-connected at boot |
| Amplifier | TPA3116D2 |
| Speakers | Visaton FRS 13 (8 ohm) |
| Power | Single power strip for all components |

### Wiring

```
Camera USB ──→ Jetson USB port
Jetson (PulseAudio) ──→ Bluetooth A2DP ──→ ZK502-C ──→ TPA3116D2 amplifier ──→ Speakers
Power strip ──→ Jetson + Amplifier
```

## MIDI modes

The `--midi-mode` flag selects how body movement maps to MIDI:

| Mode | Description | Scale/Harmony |
|---|---|---|
| `gesture` | Direction-based, 3 voices (default). Upward arm gesture = ascending notes, energy = volume, duration = articulation. | D dorian |
| `realtime` | Per-keypoint velocity-driven. Each tracked keypoint (wrists, ankles) maps to its own pitch range. | D minor pentatonic |
| `jetson` | Velocity-driven with hysteresis and sustained notes. Notes hold while movement continues. | D minor chord progression |
| `musical` | Per-frame note triggers. Original mode with silence tracking and chord harmony. | D minor chord progression |

All modes share the same detection loop and person tracking. Adding a new mode
requires only implementing a class with `update(landmarks)` and `close()`.

## Quick start (Jetson — first time)

```bash
# 1. Clone the repository
git clone <repo-url> /home/maramotto/cuerposonoro
cd /home/maramotto/cuerposonoro

# 2. Run the one-time setup script
#    Installs system packages, creates venv, downloads soundfont,
#    pre-generates the TensorRT engine (~2 min), enables the systemd service.
bash deployment/setup_jetson.sh

# 3. Start the service (or just reboot — it starts automatically)
sudo systemctl start cuerposonoro
```

After setup, the Jetson boots and runs autonomously on power. Plug in the
power strip and walk away.

## Development (Mac)

TensorRT is not available on macOS. Set `use_tensorrt: false` in `config.yaml`
before running locally.

```bash
# Clone and set up
git clone <repo-url>
cd cuerposonoro-jetson
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Edit config for Mac
# In config.yaml, change: use_tensorrt: false

# Run tests (no hardware needed)
pytest tests/unit/ -v

# Run the application (requires camera and Fluidsynth)
# Install Fluidsynth on Mac: brew install fluid-synth
python main.py
```

### Command-line flags

```bash
python main.py --mode midi --midi-mode gesture       # defaults
python main.py --midi-mode musical                    # per-frame note triggers
python main.py --midi-mode realtime                   # per-keypoint velocity
python main.py --midi-mode jetson                     # sustained notes
python main.py --config path/to/other_config.yaml     # custom config
```

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `midi` | Audio engine mode |
| `--midi-mode` | `gesture` | MIDI mapping strategy (see MIDI modes above) |
| `--config` | `config.yaml` | Path to configuration file |

## Deployment

### Updating code on the Jetson

```bash
# From your Mac — sync code (excludes git history and generated files)
rsync -avz --exclude='.git' --exclude='*.engine' --exclude='venv/' \
  --exclude='logs/' --exclude='__pycache__/' \
  . maramotto@jetson:/home/maramotto/cuerposonoro/

# SSH in and restart the service
ssh maramotto@jetson
sudo systemctl restart cuerposonoro
```

### Automatic startup

The installation starts automatically when the Jetson is powered on:

1. **Hardware:** The Jetson Orin Nano boots on DC power (default behaviour).
2. **Bluetooth service** (`deployment/bt-zk502.service`):
   - Runs as user `maramotto` at login
   - Waits for BlueZ, connects ZK502-C, sets it as default PulseAudio sink
3. **Startup script** (`deployment/start_cuerposonoro.sh`):
   - Activates the Python venv
   - Waits up to 30s for the camera (`/dev/video0`)
   - Waits up to 30s for PulseAudio
   - Waits up to 60s for Bluetooth A2DP sink (falls back to default if unavailable)
   - Launches `main.py --midi-mode gesture`
4. **systemd service** (`deployment/cuerposonoro.service`):
   - Runs as user `maramotto`
   - Restarts automatically on any crash (10s delay)
   - Routes logs to both journald and `logs/cuerposonoro.log`

See `docs/deployment/autostart.md` for the full autostart guide.

### Managing the service

```bash
# Check status
sudo systemctl status cuerposonoro

# Stop
sudo systemctl stop cuerposonoro

# Restart
sudo systemctl restart cuerposonoro

# Disable autostart (temporarily)
sudo systemctl disable cuerposonoro

# Re-enable autostart
sudo systemctl enable cuerposonoro
```

### Viewing logs

```bash
# Live journal logs
journalctl -u cuerposonoro -f

# Logs since last boot
journalctl -u cuerposonoro -b

# Persistent file log
tail -f /home/maramotto/cuerposonoro/logs/cuerposonoro.log
```

## Running tests

```bash
# Unit tests only (fast, no hardware, no I/O)
pytest tests/unit/ -v

# Integration tests (mocked I/O)
pytest tests/integration/ -v

# Full suite
pytest -v
```

### Manual diagnostics

The `tests/manual/` directory contains scripts for hardware debugging on the
Jetson. These are **not** part of the automated test suite.

```bash
# Camera diagnostic — prints FPS and resolution, fully headless
python tests/manual/manual_camera.py --duration 10

# Save every 30th frame as JPEG for remote inspection
python tests/manual/manual_camera.py --save-frames 30 --duration 60

# Retrieve saved frames to your Mac
scp maramotto@jetson:~/cuerposonoro/tests/manual/logs/frames/*.jpg ./local_frames/
```

See `tests/manual/README.md` for all flags and usage examples.

## Configuration

All tunable parameters live in `config.yaml`. No magic numbers in code.

### `vision`

| Parameter | Default | Description |
|---|---|---|
| `model` | `yolov8n-pose` | YOLOv8-Pose model variant |
| `confidence_threshold` | `0.5` | Minimum detection confidence |
| `input_size` | `640` | Model input resolution (pixels) |
| `use_tensorrt` | `true` | TensorRT GPU acceleration. Set `false` on Mac. |
| `tensorrt_half` | `true` | FP16 precision on Ampere GPU (faster, negligible accuracy loss) |

**TensorRT notes:**
- The `.engine` file is generated once during `setup_jetson.sh` (~2 min) and cached on disk.
- It is hardware-specific (tied to the exact GPU, CUDA, and TensorRT version). Do not copy between machines.
- If the `.engine` file is missing at startup, it is exported automatically (with a one-time delay).
- If export fails (e.g. on Mac), the detector falls back to CPU with the `.pt` model.

### `silence`

| Parameter | Default | Description |
|---|---|---|
| `velocity_threshold` | `0.008` | Body velocity below this is considered "still" |
| `timeout_ms` | `500` | Milliseconds of stillness before both channels go silent |

### `melody` (vibraphone, MIDI channel 1)

| Parameter | Default | Description |
|---|---|---|
| `channel` | `0` | MIDI channel (0-indexed, displays as channel 1) |
| `program` | `11` | GM program number (vibraphone) |
| `note_min` | `48` | Lowest MIDI note (C3) |
| `note_max` | `84` | Highest MIDI note (C6) |
| `velocity_min` | `30` | Minimum MIDI velocity for soft arm movement |
| `velocity_max` | `120` | Maximum MIDI velocity for fast arm movement |
| `trigger_threshold` | `0.018` | Arm velocity needed to trigger a note |
| `brightness_cc` | `74` | MIDI CC number for timbre brightness (wrist separation) |
| `note_cooldown_ms` | `150` | Minimum ms between note-on events (rate limiter) |

### `bass` (acoustic bass, MIDI channel 2)

| Parameter | Default | Description |
|---|---|---|
| `channel` | `1` | MIDI channel (0-indexed, displays as channel 2) |
| `program` | `32` | GM program number (acoustic bass pizzicato) |
| `trigger_threshold` | `0.018` | Ankle velocity needed to trigger a bass note |
| `velocity` | `100` | Fixed bass note velocity |

### `harmony`

| Parameter | Default | Description |
|---|---|---|
| `torso_tilt_threshold` | `0.05` | Lateral torso tilt needed to advance/retreat in chord progression |
| `head_tilt_threshold` | `0.04` | Lateral head tilt needed to add tension or simplify chord |
| `chord_progression` | (see config.yaml) | 6-chord progression in D minor (Dm9 → G13sus4 → Cmaj7#11 → Fmaj9 → Bø7 → E7alt) |

### `fluidsynth`

| Parameter | Default | Description |
|---|---|---|
| `soundfont` | `JJazzLab-SoundFont.sf2` | Path to the SF2 soundfont file |
| `gain` | `2.0` | Master volume |
| `sample_rate` | `44100` | Audio sample rate in Hz |
| `driver` | `pulseaudio` | Audio driver: `coreaudio` on Mac, `pulseaudio` on Jetson |

### Mode-specific configuration

Each MIDI mode has its own config section (`gesture`, `realtime`, `jetson`).
The `musical` mode uses `melody`, `bass`, `harmony`, and `silence` sections
directly. See `config.yaml` for the full reference with inline comments.

## Project structure

```
cuerposonoro-jetson/
  CLAUDE.md               # Claude Code project context
  DESIGN.md               # Full artistic and technical design rationale
  README.md               # This file
  requirements.txt        # Python dependencies
  config.yaml             # All tunable parameters

  main.py                 # Entry point — shared detection loop, mode dispatch

  vision/
    detector.py           # YOLOv8-Pose wrapper (TensorRT or CPU)
    landmarks.py          # Landmark extraction, normalisation, velocity
    capture.py            # Webcam wrapper (cv2.VideoCapture)

  features/
    arms.py               # Melody descriptors (wrist height, arm velocity, brightness)
    legs.py               # Rhythm descriptors (ankle velocity)
    harmony.py            # Torso/head tilt → chord progression control
    silence.py            # Silence detection (velocity below threshold for timeout)

  audio/
    chords.py             # Chord voicings, note selection, tension/simplification
    fluidsynth.py         # In-process Fluidsynth synth via pyfluidsynth
    midi.py               # MIDI note/CC output via pyfluidsynth direct API
    platform.py           # Factory for platform-aware Fluidsynth setup
    musical_mode.py       # Musical mode: per-frame triggers, chord progression
    gesture_mode.py       # Gesture mode: direction-based, 3 voices, Re dorian
    realtime_mode.py      # Realtime mode: per-keypoint velocity, D minor pentatonic
    jetson_sender.py      # Jetson mode: velocity-driven, sustained notes

  deployment/
    cuerposonoro.service  # systemd unit file (auto-start, watchdog)
    start_cuerposonoro.sh # Startup script (venv, device waits, logging)
    setup_jetson.sh       # One-time Jetson setup (packages, venv, soundfont, TensorRT)
    bt-zk502.service      # systemd user service for Bluetooth A2DP auto-connect
    bt-zk502-connect.sh   # Bluetooth connection script (ZK502-C speaker)

  docs/
    deployment/
      autostart.md        # Full autostart guide with troubleshooting

  tests/
    unit/                 # Fast tests, no hardware (pytest)
    integration/          # Mocked I/O tests (pytest)
    manual/               # Headless hardware diagnostics (not run by pytest)
```

## Troubleshooting

### No sound

1. Check Fluidsynth is running: `journalctl -u cuerposonoro -b | grep -i fluid`
2. Check PulseAudio: `sudo -u maramotto pactl list sinks short`
3. Check Bluetooth speaker: `bluetoothctl info E2:70:F5:E3:73:FC`
4. Check the default sink: `sudo -u maramotto pactl get-default-sink`

### Camera not found

1. Check the camera is connected: `ls /dev/video*`
2. Check device details: `v4l2-ctl --list-devices`
3. The startup script waits 30s for `/dev/video0` — if it times out, the service restarts after 10s

### Bluetooth speaker not connecting

1. Check BlueZ is powered on: `bluetoothctl show | grep Powered`
2. Check device is trusted: `bluetoothctl info E2:70:F5:E3:73:FC | grep Trusted`
3. Check the Bluetooth service: `systemctl --user status bt-zk502`
4. The startup script continues with the default sink if Bluetooth is unavailable after 60s

### Service keeps restarting

```bash
# See what went wrong
journalctl -u cuerposonoro -b --no-pager | tail -50
```

Common causes: missing camera, PulseAudio not starting, missing soundfont, corrupt TensorRT engine.

### TensorRT engine issues

If the `.engine` file is corrupt or was built for a different TensorRT version:

```bash
rm /home/maramotto/cuerposonoro/yolov8n-pose.engine
sudo systemctl restart cuerposonoro
# The engine will be re-exported automatically (~2 min delay)
```

## Credits and licences

- **JJazzLab SoundFont SF2:** based on SGM-v2.01-NicePianosGuitarsBass by John
  Nebauer. Free to use, attribution required.
- **YOLOv8-Pose:** Ultralytics, AGPL-3.0 licence.
- **Fluidsynth:** LGPL-2.1 licence.
- **pyfluidsynth:** LGPL-2.1 licence.
- **OpenCV:** Apache 2.0 licence.
