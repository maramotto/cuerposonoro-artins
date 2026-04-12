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

# --- Duplicate output to log file AND stdout (journald) ---
exec > >(tee -a "${LOG_FILE}") 2>&1

# --- Launch CuerpoSonoro ---
cd "${PROJECT_DIR}"
echo "Starting CuerpoSonoro at $(date -Iseconds)"
exec python main.py --mode midi --midi-mode musical
