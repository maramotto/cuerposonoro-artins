#!/usr/bin/env bash
# setup_jetson.sh — One-time setup script for the Jetson Orin Nano
# Run this script once after cloning the repository on the Jetson.

set -e

PROJECT_DIR="/home/jetson/cuerposonoro-jetson"
SOUNDFONT_URL="https://archive.org/download/jjazz-lab-sound-font/JJazzLab-SoundFont.sf2"
SOUNDFONT_PATH="${PROJECT_DIR}/JJazzLab-SoundFont.sf2"
SERVICE_FILE="${PROJECT_DIR}/deployment/cuerposonoro.service"

echo "=== Installing system packages ==="
sudo apt-get update
sudo apt-get install -y fluidsynth libasound2-dev alsa-utils python3-pip libopencv-dev

echo "=== Installing Python packages ==="
pip3 install -r "${PROJECT_DIR}/requirements.txt"

echo "=== Downloading JJazzLab SoundFont ==="
if [ -f "${SOUNDFONT_PATH}" ]; then
    echo "SoundFont already present at ${SOUNDFONT_PATH}, skipping download."
else
    echo "Downloading SoundFont from ${SOUNDFONT_URL} ..."
    wget -O "${SOUNDFONT_PATH}" "${SOUNDFONT_URL}"
    echo "SoundFont downloaded successfully."
fi

echo "=== Installing and enabling systemd service ==="
sudo cp "${SERVICE_FILE}" /etc/systemd/system/cuerposonoro.service
sudo systemctl daemon-reload
sudo systemctl enable cuerposonoro.service

echo "=== Setup complete ==="
echo "The service is enabled and will start on next boot."
echo "To start it now, run: sudo systemctl start cuerposonoro.service"
