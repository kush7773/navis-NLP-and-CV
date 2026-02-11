#!/bin/bash
# Setup script for Navis Robot on Ubuntu/Raspberry Pi

echo "🤖 =================================================="
echo "   Navis Robot - Complete Setup Script"
echo "   =================================================="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo "⚠️  Please don't run as root (no sudo needed for this script)"
    exit 1
fi

echo "📦 Step 1: Updating system packages..."
sudo apt update
sudo apt upgrade -y

echo ""
echo "📦 Step 2: Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-dev \
    portaudio19-dev \
    python3-pyaudio \
    espeak \
    ffmpeg \
    libportaudio2 \
    libasound2-dev \
    flac

echo ""
echo "📦 Step 3: Installing Python packages..."
pip3 install --upgrade pip

# Install all required packages
pip3 install \
    flask \
    requests \
    SpeechRecognition \
    pyttsx3 \
    pyaudio \
    vosk \
    opencv-python \
    pyserial \
    pydub \
    numpy

# RPi.GPIO (only on Raspberry Pi)
if [ -f /proc/device-tree/model ]; then
    if grep -q "Raspberry Pi" /proc/device-tree/model; then
        echo "🍓 Detected Raspberry Pi - Installing GPIO library..."
        pip3 install RPi.GPIO
    fi
fi

echo ""
echo "📦 Step 4: Creating directories..."
mkdir -p templates
mkdir -p model

echo ""
echo "📝 Step 5: Checking Vosk model..."
if [ ! -d "model" ] || [ -z "$(ls -A model)" ]; then
    echo "⚠️  Vosk model not found!"
    echo "📥 Downloading small English model (~40MB)..."
    wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
    unzip vosk-model-small-en-us-0.15.zip
    mv vosk-model-small-en-us-0.15 model
    rm vosk-model-small-en-us-0.15.zip
    echo "✅ Vosk model installed!"
else
    echo "✅ Vosk model already exists"
fi

echo ""
echo "🔧 Step 6: Setting up configuration..."
if [ ! -f "config.py" ]; then
    echo "⚠️  config.py not found! Please create it first."
else
    echo "✅ config.py found"
fi

echo ""
echo "🔑 Step 7: API Key Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "You need to get FREE API keys:"
echo ""
echo "1️⃣  GROQ API (Recommended - Ultra Fast)"
echo "   → Visit: https://console.groq.com"
echo "   → Sign up (free)"
echo "   → Get API key from dashboard"
echo "   → Free tier: Very generous limits"
echo ""
echo "2️⃣  PERPLEXITY API (Optional - Has Internet Access)"
echo "   → Visit: https://www.perplexity.ai/settings/api"
echo "   → Sign up (free)"
echo "   → Get API key"
echo "   → Free tier: Limited but available"
echo ""
echo "After getting your keys, edit config.py:"
echo "   nano config.py"
echo ""
echo "Replace:"
echo "   GROQ_API_KEY = 'YOUR_GROQ_API_KEY_HERE'"
echo "   PERPLEXITY_API_KEY = 'YOUR_PERPLEXITY_API_KEY_HERE'"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "🌐 Step 8: Network Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Your Raspberry Pi IP address:"
hostname -I | awk '{print "   📍 " $1}'
echo ""
echo "To access from your phone:"
echo "   1. Connect phone to same WiFi as Raspberry Pi"
echo "   2. Open browser on phone"
echo "   3. Go to: http://$(hostname -I | awk '{print $1}'):5001"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "✅ =================================================="
echo "   Setup Complete!"
echo "   =================================================="
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Edit config.py and add your API keys:"
echo "   nano config.py"
echo ""
echo "2. Test the LLM handler:"
echo "   python3 llm_handler.py"
echo ""
echo "3. Run the web voice interface:"
echo "   python3 web_voice_interface.py"
echo ""
echo "4. Or run the standalone voice mode:"
echo "   python3 main_updated.py"
echo ""
echo "🎉 Happy robot building!"
echo ""
