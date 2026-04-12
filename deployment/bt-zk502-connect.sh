#!/bin/bash
# bt-zk502-connect.sh — Connect ZK502-C via Bluetooth and set as default sink.
# Called by the bt-zk502 user service at login.

LOG="systemd-cat -t bt-zk502"
MAC="E2:70:F5:E3:73:FC"
SINK_PATTERN="E2_70_F5_E3_73_FC"

# Wait for BlueZ to be powered on
echo "Waiting for BlueZ..." | $LOG
until bluetoothctl show | grep -q "Powered: yes"; do
    sleep 2
done
echo "BlueZ powered on" | $LOG

# Wait for auto-connect (trusted+bonded devices reconnect automatically)
echo "Waiting for ZK502-C auto-connect..." | $LOG
for i in $(seq 1 15); do
    if bluetoothctl info "$MAC" 2>/dev/null | grep -q "Connected: yes"; then
        echo "ZK502-C connected (auto)" | $LOG
        break
    fi
    sleep 2
done

# If not connected, force it
if ! bluetoothctl info "$MAC" 2>/dev/null | grep -q "Connected: yes"; then
    echo "Forcing connection..." | $LOG
    bluetoothctl connect "$MAC"
    sleep 3
fi

# Verify connection
if ! bluetoothctl info "$MAC" 2>/dev/null | grep -q "Connected: yes"; then
    echo "ERROR: ZK502-C not connected" | $LOG
    exit 1
fi

# Wait for the PipeWire/WirePlumber sink to appear
echo "Waiting for A2DP sink..." | $LOG
for i in $(seq 1 20); do
    SINK=$(pactl list sinks short 2>/dev/null | grep -i "$SINK_PATTERN" | awk '{print $2}')
    if [ -n "$SINK" ]; then
        pactl set-default-sink "$SINK"
        echo "Default sink set: $SINK" | $LOG
        exit 0
    fi
    sleep 2
done

echo "ERROR: A2DP sink not available after 40s" | $LOG
exit 1
