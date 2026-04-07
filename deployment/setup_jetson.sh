#!/usr/bin/env bash
# setup_jetson.sh — One-time setup script for the Jetson Orin Nano.
# Run this script once after cloning the repository on the Jetson.
# Usage: bash deployment/setup_jetson.sh

set -euo pipefail

PROJECT_DIR="/home/maramotto/cuerposonoro"
VENV_DIR="${PROJECT_DIR}/venv"
SOUNDFONT_URL="https://archive.org/download/jjazz-lab-sound-font/JJazzLab-SoundFont.sf2"
SOUNDFONT_PATH="${PROJECT_DIR}/JJazzLab-SoundFont.sf2"
SERVICE_SRC="${PROJECT_DIR}/deployment/cuerposonoro.service"
SERVICE_DST="/etc/systemd/system/cuerposonoro.service"

echo "=== Installing system packages ==="
sudo apt-get update
sudo apt-get install -y \
    fluidsynth \
    libasound2-dev \
    alsa-utils \
    pulseaudio-utils \
    python3-pip \
    python3-venv \
    libopencv-dev

echo "=== Creating virtual environment ==="
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo "Virtual environment created at ${VENV_DIR}"
else
    echo "Virtual environment already exists at ${VENV_DIR}, skipping."
fi

echo "=== Installing Python packages ==="
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install -r "${PROJECT_DIR}/requirements.txt"

echo "=== Creating log directory ==="
mkdir -p "${PROJECT_DIR}/logs"

echo "=== Downloading JJazzLab SoundFont ==="
if [ -f "${SOUNDFONT_PATH}" ]; then
    echo "SoundFont already present at ${SOUNDFONT_PATH}, skipping download."
else
    echo "Downloading SoundFont from ${SOUNDFONT_URL} ..."
    wget -O "${SOUNDFONT_PATH}" "${SOUNDFONT_URL}"
    echo "SoundFont downloaded successfully."
fi

echo "=== Pre-generating TensorRT engine (this takes ~2 minutes) ==="
cd "${PROJECT_DIR}"
python -c "from vision.detector import PoseDetector; PoseDetector('yolov8n-pose.pt', confidence=0.5, use_tensorrt=True)"
echo "TensorRT engine ready."

echo "=== Making startup script executable ==="
chmod +x "${PROJECT_DIR}/deployment/start_cuerposonoro.sh"

echo "=== Installing and enabling systemd service ==="
sudo cp "${SERVICE_SRC}" "${SERVICE_DST}"
sudo systemctl daemon-reload
sudo systemctl enable cuerposonoro.service

echo ""
echo "=== Setup complete ==="
echo "The service is enabled and will start on next boot."
echo ""
echo "To start it now:    sudo systemctl start cuerposonoro"
echo "To check status:    sudo systemctl status cuerposonoro"
echo "To view live logs:  journalctl -u cuerposonoro -f"
echo "To view file logs:  tail -f ${PROJECT_DIR}/logs/cuerposonoro.log"
