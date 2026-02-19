#!/bin/bash

# NAVIS Robot - Complete Control Launch Script
# Runs the unified control interface

BLUE='\033[0;34m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${CYAN}═══════════════════════════════════════════════${NC}"
echo -e "${CYAN}   🤖 NAVIS COMPLETE CONTROL INTERFACE        ${NC}"
echo -e "${CYAN}   Robomanthan × BNMIT                        ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════${NC}"

# Check dependencies
echo -e "\n${YELLOW}[1/3] Checking dependencies...${NC}"

# Check Python
if command -v python3 &> /dev/null; then
    echo -e "${GREEN}  ✓ Python3 found${NC}"
else
    echo -e "${RED}  ✗ Python3 not found! Please install Python 3.${NC}"
    exit 1
fi

# Check required packages
MISSING=0
for pkg in flask cv2 speech_recognition pydub face_recognition numpy requests; do
    if ! python3 -c "import $pkg" 2>/dev/null; then
        echo -e "${RED}  ✗ Missing: $pkg${NC}"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo -e "${YELLOW}  Installing missing packages...${NC}"
    pip3 install flask opencv-python speech_recognition pydub face_recognition numpy requests pyserial
fi

echo -e "${GREEN}  ✓ All dependencies OK${NC}"

# Check hardware
echo -e "\n${YELLOW}[2/3] Checking hardware...${NC}"

# Camera
if python3 -c "import cv2; c=cv2.VideoCapture(0); r,_=c.read(); c.release(); exit(0 if r else 1)" 2>/dev/null; then
    echo -e "${GREEN}  ✓ Camera: ONLINE${NC}"
else
    echo -e "${YELLOW}  ⚠ Camera: NOT DETECTED (interface will run without camera)${NC}"
fi

# Serial/ESP32
if ls /dev/ttyUSB* 1> /dev/null 2>&1 || ls /dev/ttyACM* 1> /dev/null 2>&1; then
    echo -e "${GREEN}  ✓ ESP32/Arduino: ONLINE${NC}"
else
    echo -e "${YELLOW}  ⚠ ESP32/Arduino: NOT DETECTED (motor control disabled)${NC}"
fi

# Detect IP
echo -e "\n${YELLOW}[3/3] Starting NAVIS...${NC}"
if command -v hostname &> /dev/null; then
    IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    if [ -z "$IP" ]; then
        IP="localhost"
    fi
else
    IP="localhost"
fi

echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}  📲 Access URL: http://${IP}:5000            ${NC}"
echo -e "${GREEN}  📲 Local URL:  http://localhost:5000        ${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo ""

# Launch the application
python3 "${SCRIPT_DIR}/navis_complete_control.py"
