# cuerposonoro-jetson

An autonomous art installation that translates full-body human movement into
real-time sound. Built for the NVIDIA Jetson Orin Nano — boots and runs on power
alone, no screen, no manual intervention.

This is a separate project from [cuerposonoro](https://github.com/mara/cuerposonoro)
(the main thesis). It adapts the body-to-sound mapping for a specific physical
context with conscious technical trade-offs. See `DESIGN.md` for the full
artistic and technical rationale.

## How it works

A camera watches the space. YOLOv8-Pose detects people and their body landmarks.
Movement is analysed in real time:

- **Arms** control a vibraphone melody (MIDI channel 1)
- **Legs** control an acoustic bass rhythm (MIDI channel 2)
- **Torso tilt** advances or retreats through a chord progression
- **Head tilt** adds tension or simplifies the active chord
- **No movement = no sound.** Silence is part of the work.

Multiple people can interact with the installation simultaneously.

## Hardware requirements

| Component | Specification |
|---|---|
| Compute | NVIDIA Jetson Orin Nano, JetPack 6.1 |
| Camera | Logitech C922 (USB) |
| Audio | TPA3116D2 amplifier + Visaton FRS 13 speakers (8 ohm) |
| Power | Single power strip for all components |

### Wiring

1. Camera → Jetson USB port
2. Jetson 3.5mm audio out → TPA3116D2 amplifier input
3. Amplifier output → Visaton FRS 13 speakers
4. All powered from one power strip

## One-time Jetson setup

```bash
# Clone the repo
git clone <repo-url> ~/cuerposonoro-jetson
cd ~/cuerposonoro-jetson

# Run the setup script
bash deployment/setup_jetson.sh
```

The setup script installs system dependencies (Fluidsynth, ALSA tools),
Python packages, downloads the soundfont, and installs the systemd service.

## Development (Mac)

```bash
# Clone and enter the project
git clone <repo-url>
cd cuerposonoro-jetson

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/unit/ -v

# Run the application (requires camera and Fluidsynth)
python main.py
```

Note: on Mac, Fluidsynth uses CoreAudio instead of ALSA. Install it with
`brew install fluid-synth`. The MIDI virtual port works natively on macOS.

## Deployment to Jetson

```bash
# From your Mac
rsync -avz --exclude='.git' . jetson:~/cuerposonoro-jetson/

# SSH into the Jetson and restart the service
ssh jetson
sudo systemctl restart cuerposonoro
```

## Running tests

```bash
# Unit tests only (fast, no hardware)
pytest tests/unit/ -v

# Integration tests (mocked I/O)
pytest tests/integration/ -v

# All tests
pytest -v
```

## Configuration

All tunable parameters are in `config.yaml`. No magic numbers in code.

### `vision`

| Parameter | Default | Description |
|---|---|---|
| `model` | `yolov8n-pose` | YOLOv8-Pose model variant |
| `confidence_threshold` | `0.5` | Minimum detection confidence |
| `input_size` | `640` | Model input resolution |

### `silence`

| Parameter | Default | Description |
|---|---|---|
| `velocity_threshold` | `0.02` | Body velocity below this = "still" |
| `timeout_ms` | `500` | Duration of stillness before silence kicks in |

### `melody` (vibraphone, MIDI channel 1)

| Parameter | Default | Description |
|---|---|---|
| `channel` | `0` | MIDI channel (0-indexed) |
| `program` | `11` | GM program number (vibraphone) |
| `note_min` | `48` | Lowest playable MIDI note (C3) |
| `note_max` | `84` | Highest playable MIDI note (C6) |
| `velocity_min` | `30` | Minimum MIDI velocity |
| `velocity_max` | `120` | Maximum MIDI velocity |
| `trigger_threshold` | `0.03` | Arm velocity needed to trigger a note |
| `brightness_cc` | `74` | MIDI CC number for brightness control |

### `bass` (acoustic bass, MIDI channel 2)

| Parameter | Default | Description |
|---|---|---|
| `channel` | `1` | MIDI channel (0-indexed) |
| `program` | `32` | GM program number (acoustic bass) |
| `trigger_threshold` | `0.03` | Ankle velocity needed to trigger a note |
| `velocity` | `100` | Fixed bass velocity |

### `harmony`

| Parameter | Default | Description |
|---|---|---|
| `torso_tilt_threshold` | `0.05` | Tilt needed to change chord |
| `head_tilt_threshold` | `0.04` | Tilt needed to modify chord tension |
| `chord_progression` | (see config.yaml) | 6-chord progression in D minor |

### `fluidsynth`

| Parameter | Default | Description |
|---|---|---|
| `soundfont` | `JJazzLab-SoundFont.sf2` | Path to the SF2 soundfont |
| `gain` | `0.8` | Master volume (0.0–1.0) |
| `sample_rate` | `44100` | Audio sample rate in Hz |

### `midi`

| Parameter | Default | Description |
|---|---|---|
| `port_name` | `cuerposonoro` | Name of the virtual MIDI port |

## Credits and licences

- **JJazzLab SoundFont SF2:** based on SGM-v2.01-NicePianosGuitarsBass by John
  Nebauer. Free to use, attribution required.
- **YOLOv8-Pose:** Ultralytics, AGPL-3.0 licence.
- **Fluidsynth:** LGPL-2.1 licence.
- **mido:** MIT licence.
