#!/usr/bin/env bash
# start_cuerposonoro.sh — Startup script for the CuerpoSonoro installation.
# Called by the cuerposonoro.service systemd unit.
# Activates the venv, waits for hardware, and launches main.py.

set -euo pipefail

PROJECT_DIR="/home/maramotto/cuerposonoro-artins/cuerposonoro-artins"
VENV_DIR="${PROJECT_DIR}/venv"
LOG_DIR="${PROJECT_DIR}/logs"
LOG_FILE="${LOG_DIR}/cuerposonoro.log"
CAMERA_DEVICE="/dev/video0"
CAMERA_TIMEOUT=30
PULSE_TIMEOUT=30

# --- Ensure log directory exists ---
mkdir -p "${LOG_DIR}"

# --- Activate virtual environment ---
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

# --- Wait for camera ---
echo "Waiting for ${CAMERA_DEVICE} (timeout: ${CAMERA_TIMEOUT}s)..."
elapsed=0
while [ ! -e "${CAMERA_DEVICE}" ]; do
    if [ "${elapsed}" -ge "${CAMERA_TIMEOUT}" ]; then
        echo "ERROR: ${CAMERA_DEVICE} not available after ${CAMERA_TIMEOUT}s" >&2
        exit 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
done
echo "Camera ready: ${CAMERA_DEVICE}"

# --- Wait for PulseAudio ---
echo "Waiting for PulseAudio (timeout: ${PULSE_TIMEOUT}s)..."
elapsed=0
while ! pactl info >/dev/null 2>&1; do
    if [ "${elapsed}" -ge "${PULSE_TIMEOUT}" ]; then
        echo "ERROR: PulseAudio not ready after ${PULSE_TIMEOUT}s" >&2
        exit 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
done
echo "PulseAudio ready"

# --- Wait for Bluetooth A2DP sink (ZK502-C) ---
BT_SINK_PATTERN="E2_70_F5_E3_73_FC"
BT_TIMEOUT=60
echo "Waiting for Bluetooth sink (timeout: ${BT_TIMEOUT}s)..."
elapsed=0
while ! pactl list sinks short 2>/dev/null | grep -q "$BT_SINK_PATTERN"; do
    if [ "${elapsed}" -ge "${BT_TIMEOUT}" ]; then
        echo "WARNING: Bluetooth sink not available after ${BT_TIMEOUT}s, continuing with default sink" >&2
        break
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done
if pactl list sinks short 2>/dev/null | grep -q "$BT_SINK_PATTERN"; then
    BT_SINK=$(pactl list sinks short | grep "$BT_SINK_PATTERN" | awk '{print $2}')
    pactl set-default-sink "$BT_SINK"
    echo "Bluetooth sink active: $BT_SINK"
fi

# --- Duplicate output to log file AND stdout (journald) ---
exec > >(tee -a "${LOG_FILE}") 2>&1

# --- Launch CuerpoSonoro ---
cd "${PROJECT_DIR}"
echo "Starting CuerpoSonoro at $(date -Iseconds)"
exec python main.py --mode midi --midi-mode gesture
