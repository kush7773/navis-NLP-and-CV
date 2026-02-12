#!/bin/bash

# Navis Robot Startup Script
# --------------------------
# 1. Installs dependencies
# 2. Generates SSL certificates
# 3. Detects IP address
# 4. Runs the application

BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🤖 Starting Navis Setup...${NC}"

# 1. Install Dependencies
echo -e "\n${BLUE}📦 Checking dependencies...${NC}"
if ! pip3 show pyopenssl > /dev/null 2>&1; then
    echo "Installing dependencies..."
    sudo apt-get update && sudo apt-get install -y flac
    # install opencv-contrib-python for cv2.face module
    pip3 install pyopenssl flask opencv-contrib-python face_recognition speechrecognition pydub
else
    echo -e "${GREEN}✓ Dependencies installed${NC}"
fi

# 2. SSL Certificates
echo -e "\n${BLUE}🔐 Checking SSL Certificates...${NC}"
if [[ ! -f "cert.pem" || ! -f "key.pem" ]]; then
    echo "Generating self-signed certificate..."
    openssl req -new -newkey rsa:2048 -days 365 -nodes -x509 \
        -keyout key.pem -out cert.pem \
        -subj "/C=US/ST=State/L=City/O=Navis/CN=raspberrypi" >/dev/null 2>&1
    echo -e "${GREEN}✓ Certificates generated${NC}"
else
    echo -e "${GREEN}✓ Certificates exist${NC}"
fi

# 3. Detect IP
echo -e "\n${BLUE}🌐 Detecting IP Address...${NC}"
IP=$(hostname -I | awk '{print $1}')
echo -e "Device IP: ${GREEN}$IP${NC}"

# 4. Run Application
echo -e "\n${BLUE}🚀 Starting Navis Control Interface...${NC}"
echo -e "---------------------------------------------------"
echo -e "📲 Access URL: ${GREEN}https://$IP:5000${NC}"
echo -e "⚠️  Accept the 'Not Secure' warning in browser"
echo -e "---------------------------------------------------"

# Update config.py dynamically (optional, but helps avoid confusion)
# sed -i "s/RASPBERRY_PI_IP = .*/RASPBERRY_PI_IP = \"$IP\"/" config.py

python3 voice_follow_integration.py
