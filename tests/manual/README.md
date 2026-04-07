# Manual diagnostic scripts

These scripts are for manual hardware diagnostics on the Jetson.
They are **not** part of the automated test suite and are never run by `pytest`.

All scripts run fully headless — no display, no `cv2.imshow()`, no Qt.

## manual_camera.py

Captures frames from the USB camera and prints live stats to the console.
Use it to verify the camera works, check real resolution, and measure FPS.

### Usage

```bash
# Run until Ctrl+C — prints FPS every second
python tests/manual/manual_camera.py

# Run for 10 seconds then stop
python tests/manual/manual_camera.py --duration 10

# Save every 30th frame as JPEG for visual inspection
python tests/manual/manual_camera.py --save-frames 30

# Full example: 60 seconds, save every 30 frames, custom output dir
python tests/manual/manual_camera.py --duration 60 --save-frames 30 --save-dir /tmp/frames

# Use a different camera device
python tests/manual/manual_camera.py --device 1
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `0` | Camera device index (`/dev/videoN`) |
| `--duration` | `0` | Stop after N seconds (0 = run until Ctrl+C) |
| `--save-frames` | `0` | Save every Nth frame as JPEG (0 = disabled) |
| `--save-dir` | `tests/manual/logs/frames` | Directory for saved frames |

### Output

```
Camera opened: 1920x1080 on /dev/video0
Running until Ctrl+C
[   1.0s] fps= 30.0  avg= 30.0  frames=30  failed=0
[   2.0s] fps= 29.8  avg= 29.9  frames=60  failed=0
^C
--- Summary ---
Duration:      2.3s
Total frames:  68
Failed frames: 0
Average FPS:   29.6
```

### Retrieving saved frames from the Jetson

```bash
scp jetson:~/cuerposonoro/tests/manual/logs/frames/*.jpg ./local_frames/
```
