# CuerpoSonoro — Automatic Startup Guide

This document describes how CuerpoSonoro starts automatically when the Jetson
Orin Nano is powered on, and how to manage the service.

---

## How it works

Three components make the installation fully autonomous:

1. **Hardware power-on:** The Jetson Orin Nano boots automatically when DC
   power is applied to the barrel jack. This is the default hardware behavior
   — no BIOS or software configuration is needed. Plugging the power strip
   into mains is the only action required.

2. **Startup script** (`deployment/start_cuerposonoro.sh`):
   - Activates the Python virtual environment (`venv/`)
   - Waits up to 30 seconds for the USB camera (`/dev/video0`)
   - Waits up to 30 seconds for PulseAudio to be ready
   - Launches `python main.py --mode midi --midi-mode musical`
   - Logs go to both `logs/cuerposonoro.log` (file) and journald

3. **systemd service** (`deployment/cuerposonoro.service`):
   - Starts after the graphical and sound targets
   - Runs as user `maramotto`
   - Restarts automatically on any failure (`Restart=always`, 10s delay)
   - Sets `DISPLAY=` (headless), `XDG_RUNTIME_DIR`, and `PULSE_SERVER`
     so PulseAudio works from a systemd context

---

## Initial setup

Run once after cloning the repository on the Jetson:

```bash
cd /home/maramotto/cuerposonoro
bash deployment/setup_jetson.sh
```

This installs system packages, creates the venv, downloads the soundfont,
and enables the systemd service.

---

## Managing the service

### Check status

```bash
sudo systemctl status cuerposonoro
```

### Start the service manually

```bash
sudo systemctl start cuerposonoro
```

### Stop the service

```bash
sudo systemctl stop cuerposonoro
```

### Restart the service

```bash
sudo systemctl restart cuerposonoro
```

### Disable autostart temporarily

```bash
sudo systemctl disable cuerposonoro
```

The service will not start on next boot. To re-enable:

```bash
sudo systemctl enable cuerposonoro
```

---

## Viewing logs

### journald (structured, rotated automatically)

```bash
# Live tail
journalctl -u cuerposonoro -f

# Last 100 lines
journalctl -u cuerposonoro -n 100

# Logs since last boot
journalctl -u cuerposonoro -b

# Logs from a specific time
journalctl -u cuerposonoro --since "2026-04-07 10:00:00"
```

### File log (persists across reboots)

```bash
# Live tail
tail -f /home/maramotto/cuerposonoro/logs/cuerposonoro.log

# Last 100 lines
tail -100 /home/maramotto/cuerposonoro/logs/cuerposonoro.log
```

The file log grows without automatic rotation. For long-running installations,
add a logrotate config:

```bash
sudo tee /etc/logrotate.d/cuerposonoro << 'EOF'
/home/maramotto/cuerposonoro/logs/cuerposonoro.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    copytruncate
}
EOF
```

---

## Troubleshooting

### Service fails to start

```bash
# Check what happened
journalctl -u cuerposonoro -b --no-pager

# Common causes:
# - Camera not connected → "ERROR: /dev/video0 not available after 30s"
# - PulseAudio not ready → "ERROR: PulseAudio not ready after 30s"
# - Missing venv → check that /home/maramotto/cuerposonoro/venv exists
# - Missing soundfont → re-run setup_jetson.sh
```

### PulseAudio issues

```bash
# Check if PulseAudio is running for maramotto
sudo -u maramotto pactl info

# Check the Scarlett 2i2 is recognised
sudo -u maramotto pactl list sinks short

# If PulseAudio is not running, start it manually
sudo -u maramotto pulseaudio --start
```

### Camera issues

```bash
# Check if the camera is detected
ls -la /dev/video*

# Check video device details
v4l2-ctl --list-devices
```

### After updating the service file

If you edit `deployment/cuerposonoro.service`, reload and restart:

```bash
sudo cp /home/maramotto/cuerposonoro/deployment/cuerposonoro.service \
        /etc/systemd/system/cuerposonoro.service
sudo systemctl daemon-reload
sudo systemctl restart cuerposonoro
```

---

## Architecture diagram

```
Power on
  └─→ Jetson boots (hardware default)
        └─→ systemd reaches graphical.target
              └─→ cuerposonoro.service starts
                    └─→ start_cuerposonoro.sh
                          ├─→ activate venv
                          ├─→ wait for /dev/video0
                          ├─→ wait for PulseAudio
                          └─→ python main.py
                                ├─→ YOLOv8-Pose (camera → landmarks)
                                ├─→ Feature extraction (arms, legs, harmony)
                                ├─→ MIDI output (Fluidsynth → PulseAudio)
                                └─→ Sound → Scarlett 2i2 → speakers
```

If the process crashes, systemd waits 10 seconds and restarts it automatically.
