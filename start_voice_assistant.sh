#!/bin/bash
# Start AI GLASS Voice Assistant with Bluetooth Microphone

echo "================================================"
echo "AI GLASS - Starting Voice Assistant"
echo "================================================"

# Activate virtual environment
source "/run/media/aadhiasarana/E/AI GLASS/train_env/bin/activate"

# Force Bluetooth to headset mode
echo "Setting up Bluetooth microphone..."
pactl set-card-profile bluez_card.C0_78_3E_CD_2D_61 headset-head-unit-cvsd 2>/dev/null || \
pactl set-card-profile bluez_card.C0_78_3E_CD_2D_61 headset-head-unit 2>/dev/null

sleep 1

# Set as default (must be done after profile switch)
pactl set-default-source bluez_input.C0:78:3E:CD:2D:61 2>/dev/null

sleep 0.5

# Verify setup
DEFAULT_SOURCE=$(pactl get-default-source)
if [[ "$DEFAULT_SOURCE" == *"bluez"* ]]; then
    echo "✓ Bluetooth microphone configured: $DEFAULT_SOURCE"
else
    echo "⚠️  Warning: Default source is $DEFAULT_SOURCE (not Bluetooth)"
    echo "   The assistant will use your laptop's built-in microphone"
    echo "   Speak into your laptop, not your Airdopes!"
fi

echo "✓ Starting backend..."
echo "================================================"
echo ""

# Run backend
python backend.py
