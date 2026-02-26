"""
Configuration file for Navis Robot
Store your API keys and settings here
"""

# ===== LLM API KEYS =====
# Groq API key (FREE - Ultra Fast)
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"  # Get from: https://console.groq.com

# Hugging Face API key (FREE Alternative with Internet Access)
# Get from: https://huggingface.co/settings/tokens
# Free tier: ~1000 requests/day, some models have internet access via tools
HUGGINGFACE_API_KEY = "YOUR_HF_API_KEY_HERE"  # Optional but recommended

# ===== LLM SETTINGS =====
# Choose which LLM to use: "groq" or "huggingface"
# Groq is faster, Hugging Face has more model options (both FREE)
PRIMARY_LLM = "groq"  # Change to "huggingface" for alternative models

# Groq model options (all free, ultra-fast):
# - "llama-3.3-70b-versatile" (Best quality, slower)
# - "llama-3.1-8b-instant" (Fast, good quality) ⭐ RECOMMENDED
# - "mixtral-8x7b-32768" (Good for longer context)
GROQ_MODEL = "llama-3.1-8b-instant"

# Hugging Face model options (free tier ~1000 req/day):
# - "mistralai/Mistral-7B-Instruct-v0.2" (Fast, good quality)
# - "microsoft/phi-2" (Very fast, lightweight)
# - "HuggingFaceH4/zephyr-7b-beta" (Friendly responses)
HUGGINGFACE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# ===== ROBOT SETTINGS =====
ROBOT_NAME = "Navis"
ROBOT_CREATOR = "Team Robomanthan"

# Response settings
MAX_RESPONSE_LENGTH = 100  # Maximum tokens in response (keep short for speech)
RESPONSE_TEMPERATURE = 0.7  # 0.0 = deterministic, 1.0 = creative

# Serial ports for motor controllers
SERIAL_PORT = "/dev/ttyESP32"  # ESP32 Motor Controller
MEGA_PORT = "/dev/ttyArduino"    # Arduino Mega Hand Array
SERIAL_BAUD = 115200

# Servo settings
SERVO_PIN = 12  # GPIO pin for mouth servo

# Motor speeds (must match ESP32 range: 0-255)
# ESP32 uses: BSPEED=55, SPEED=85, TURN_SPD=100
AUTO_SPEED = 85        # Speed for face tracking (matches ESP32 SPEED)
MANUAL_SPEED = 100     # Speed for web button controls (matches ESP32 TURN_SPD)
TURN_SPEED = 100       # Speed for turning (matches ESP32 TURN_SPD)
BASE_SPEED = 55        # Base/minimum speed (matches ESP32 BSPEED)

# ===== WEB INTERFACE =====
WEB_HOST = "0.0.0.0"  # Listen on all interfaces
WEB_PORT = 5001       # Main web interface port (navis_hybrid.py)
VOICE_PORT = 5001     # Voice control interface port (voice_follow_integration.py)
RASPBERRY_PI_IP = "192.168.0.182"  # Your Raspberry Pi IP address

# ===== FOLLOW MODE SETTINGS =====
# Voice commands to activate follow mode
FOLLOW_COMMANDS = ["follow me navis", "follow me", "navis follow me", "start following"]
STOP_COMMANDS = ["stop navis", "navis stop", "stop following", "stop"]

# Follow mode uses existing face tracking from navis_hybrid.py
FOLLOW_MODE_ENABLED = True  # Enable voice-activated follow mode

# ===== FOLLOW MODE TUNING =====
# Persistence: keep tracking for N frames after losing the target (prevents jitter stops)
PERSISTENCE_FRAMES = 15     # ~0.5s at 30fps before the robot gives up and stops

# EMA Smoothing: reduces oscillation in target position (0.0 = no change, 1.0 = instant)
# IMPORTANT: Haar cascade at 1/4 resolution produces ±40-80px noise.
# Keep this LOW (0.1-0.2) to filter out detection jitter.
SMOOTHING_ALPHA = 0.15      # Lower = smoother but slower to react

# Dead zone: ignore small centering errors (pixels) to prevent micro-adjustments
# This MUST be larger than the detection noise floor (~60px for Haar at 0.25x)
DEAD_ZONE = 70              # Pixels of error to ignore (prevents jitter turning)

# Hysteresis band: once turning, only stop when error drops below this (prevents flip-flop)
HYSTERESIS_ZONE = 30        # Must be < DEAD_ZONE. Robot keeps turning until error < this

# Motor command rate limit: minimum frames between direction changes
MIN_FRAMES_BETWEEN_DIR_CHANGE = 5  # Prevents rapid LEFT→RIGHT→LEFT oscillation

# ===== SPEECH SETTINGS =====
# Vosk model path (for offline speech recognition)
VOSK_MODEL_PATH = "model"

# TTS settings
TTS_RATE = 150        # Speech rate (words per minute)
TTS_VOLUME = 1.0      # Volume (0.0 to 1.0)
