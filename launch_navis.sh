#!/bin/bash

# --- CONFIGURATION ---
VENV_PATH="$HOME/RoboGPT/venv/bin/activate"
PROJECT_PATH="$HOME/RoboGPT"

# --- COLORS ---
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# --- SETUP DISPLAY ---
export DISPLAY=:0
xhost + > /dev/null 2>&1

# --- HEADER ---
clear
echo -e "${CYAN}=========================================${NC}"
echo -e "${CYAN}   ? NAVIS ROBOT MISSION CONTROL ?   ${NC}"
echo -e "${CYAN}=========================================${NC}"

# 1. HARDWARE CHECKS
echo -e "${YELLOW}[1/3] Checking Hardware...${NC}"

# A. Servo Daemon
if ! pgrep -x "pigpiod" > /dev/null; then
    echo "  > Starting Servo Daemon..."
    sudo pigpiod
else
    echo -e "${GREEN}  > Servo System:   ONLINE${NC}"
fi

# B. Audio/Mic
BT_CARD=$(pactl list cards short | grep bluez_card | cut -f2)
if [ ! -z "$BT_CARD" ]; then
    pactl set-card-profile $BT_CARD headset_head_unit
    echo -e "${GREEN}  > Microphone:     ONLINE (Bluetooth)${NC}"
else
    echo -e "${YELLOW}  > Microphone:     DEFAULT (Check Input)${NC}"
fi

# C. ESP32 Bridge Check (Updated for your USB setup)
if ls /dev/ttyUSB* 1> /dev/null 2>&1; then
    echo -e "${GREEN}  > ESP32 Muscle:   ONLINE${NC}"
else
    echo -e "${RED}  > ESP32 Muscle:   OFFLINE (Check USB cable!)${NC}"
fi

# 2. ACTIVATE BRAIN
echo -e "${YELLOW}[2/3] Activating Neural Network...${NC}"
source $VENV_PATH
cd $PROJECT_PATH

# 3. MISSION SELECTION
echo -e "${YELLOW}[3/3] Select Mission Profile:${NC}"
echo -e "  [1] ? Voice Only"
echo -e "  [2] ? Vision Only (Web Upload + Tracking)"
echo -e "  [3] ? FULL SYSTEM (Voice + Vision + Motors)"
echo -e "  [4] ? Exit"
echo ""
read -p "Enter Choice [1-4]: " choice

echo ""
echo -e "${CYAN}Starting Systems...${NC}"

case $choice in
    1)
        # Voice Only
        python3 main.py
        ;;
    2)
        # Vision Only (Foreground) - Uses the NEW Hybrid Brain
        echo -e "${GREEN}>> Starting Web Interface...${NC}"
        echo -e "${GREEN}>> Connect to: http://navis.local:5000${NC}"
        python3 navis_hybrid.py
        ;;
    3)
        # FULL SYSTEM (Vision Background + Voice Foreground)
        echo -e "${GREEN}>> Starting Hybrid Vision (Background)...${NC}"
        
        # Run Hybrid Brain in background, hide output so it doesn't mess up chat
        python3 navis_hybrid.py > /dev/null 2>&1 &
        WEB_PID=$!
        
        echo -e "${GREEN}>> Web Interface Active: http://navis.local:5000${NC}"
        echo -e "${YELLOW}>> Waiting 5s for Camera...${NC}"
        sleep 5
        
        echo -e "${GREEN}>> Launching Voice Core...${NC}"
        python3 main.py
        
        # Cleanup when Voice Node exits
        echo -e "${RED}>> Shutting down Systems...${NC}"
        kill $WEB_PID
        ;;
    4)
        echo "System Offline."
        exit 0
        ;;
    *)
        echo "Invalid Selection."
        ;;
esac
