# Navis Robot - NLP & Computer Vision

🤖 Advanced humanoid robot with AI conversation, voice control, computer vision, and autonomous following capabilities.

## ✨ Features

### 🎤 Voice Control
- Voice-activated commands via phone microphone
- "Follow me Navis" - Activate camera-based human tracking
- "Stop Navis" - Stop following
- Natural language conversation via Groq AI

### 📹 Computer Vision
- Live camera feed with face detection
- Real-time human tracking and following
- Bounding box visualization
- Toggle face detection on/off

### 🕹️ Manual Control
- Professional web-based joystick interface
- Touch-friendly controls (F/B/L/R/S)
- Emergency stop button
- Responsive design (mobile + desktop)

### 🤖 AI Integration
- **Groq API** - Ultra-fast, free LLM (primary)
- **Hugging Face** - Free alternative with 100+ models
- Natural conversation capabilities
- Voice-to-text and text-to-speech

### 🎨 Professional UI
- Dark cyberpunk theme
- Real-time status indicators
- Glowing animations
- All-in-one control center

## 🏗️ Architecture

```
Raspberry Pi (Master)
├── Python Control Programs
│   ├── navis_complete_control.py (Main interface)
│   ├── navis_hybrid.py (Camera & CV)
│   └── voice_follow_integration.py (Voice control)
├── serial_bridge.py (Communication)
└── ESP32 (Slave) via USB Serial
    ├── Motor Controller (Arduino)
    └── GPIO → Motor Driver → DC Motors
```

## 🚀 Quick Start

### 1. Hardware Setup
- Raspberry Pi 4 (or 3B+)
- ESP32 microcontroller
- Motor driver (L298N/BTS7960)
- 2x DC motors
- Camera module
- Servo motor (for mouth animation)

### 2. Install Dependencies

```bash
cd /path/to/navis-NLP-and-CV
chmod +x setup_navis.sh
./setup_navis.sh
```

### 3. Configure

Edit `config.py`:
```python
GROQ_API_KEY = "your_groq_api_key_here"
RASPBERRY_PI_IP = "your_pi_ip_address"
SERIAL_PORT = "/dev/ttyUSB0"  # Your ESP32 port
```

### 4. Upload ESP32 Code

1. Open Arduino IDE
2. Install ESP32 board support
3. Open `esp32_motor_controller/esp32_motor_controller.ino`
4. Upload to ESP32

### 5. Run

```bash
python3 navis_complete_control.py
```

Access from phone: `http://your_pi_ip:5000`

## 📁 Project Structure

```
navis-NLP-and-CV/
├── navis_complete_control.py    # Main control interface
├── navis_hybrid.py               # Camera & face tracking
├── voice_follow_integration.py   # Voice-activated following
├── serial_bridge.py              # ESP32 communication
├── llm_handler_updated.py        # Groq + HuggingFace LLM
├── tts.py                        # Text-to-speech
├── stt.py                        # Speech-to-text
├── servo_mouth.py                # Mouth animation
├── config.py                     # Configuration
├── templates/
│   └── complete_control.html     # Professional UI
├── esp32_motor_controller/
│   └── esp32_motor_controller.ino # ESP32 Arduino code
├── requirements.txt              # Python dependencies
├── setup_navis.sh               # Auto-setup script
├── QUICKSTART.md                # Quick start guide
├── ESP32_SETUP_GUIDE.md         # ESP32 setup instructions
├── CONTROL_INTERFACE_GUIDE.md   # UI guide
├── TESTING_GUIDE.md             # Testing instructions
└── README.md                    # This file
```

## 🎮 Controls

### Joystick Commands
- **↑ (F)** - Forward
- **↓ (B)** - Backward
- **← (L)** - Turn left
- **→ (R)** - Turn right
- **Center (S)** - Stop
- **🛑 Emergency Stop** - Instant stop

### Voice Commands
- "Follow me Navis" - Activate tracking
- "Stop Navis" - Stop following
- Any question - AI responds

## 🔧 Hardware Connections

### ESP32 Pins
```
GPIO 32 → Left Motor Forward (RPWM)
GPIO 33 → Left Motor Backward (LPWM)
GPIO 25 → Right Motor Forward (RPWM)
GPIO 26 → Right Motor Backward (LPWM)
```

### Raspberry Pi
```
GPIO 12 → Servo Motor (mouth animation)
USB → ESP32 (serial communication)
Camera → CSI/USB port
```

## 🧪 Testing

Run component tests:
```bash
python3 test_robot.py
```

This tests:
- Config loading
- ESP32 connection
- Motor control
- Groq API
- Camera
- TTS/STT
- Flask server

## 📡 Communication Protocol

**Raspberry Pi → ESP32**:
```
Format: "LEFT,RIGHT\n"
Example: "100,-100\n" (turn right)
Range: -255 to 255
```

## 🎯 API Keys

### Groq (Required)
1. Sign up: https://console.groq.com
2. Get API key
3. Add to `config.py`

### Hugging Face (Optional)
1. Sign up: https://huggingface.co
2. Get token: https://huggingface.co/settings/tokens
3. Add to `config.py`

## 🐛 Troubleshooting

### ESP32 Not Found
```bash
ls /dev/ttyUSB* /dev/ttyACM*
# Update SERIAL_PORT in config.py
```

### Motors Don't Move
- Check power supply
- Verify pin connections
- Test ESP32 with Serial Monitor
- Check motor driver enable pin

### Camera Not Working
```bash
python3 -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### API Errors
- Verify API key in `config.py`
- Check internet connection
- Test with: `python3 -c "from llm_handler_updated import get_llm; print(get_llm().ask('Hello'))"`

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide
- **[ESP32_SETUP_GUIDE.md](ESP32_SETUP_GUIDE.md)** - ESP32 setup
- **[CONTROL_INTERFACE_GUIDE.md](CONTROL_INTERFACE_GUIDE.md)** - UI guide
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Testing guide

## 🎨 Features Showcase

### Professional UI
- Dark cyberpunk theme with blue/purple gradients
- Real-time camera feed with face detection
- Virtual joystick with touch support
- Voice control with hold-to-record
- Emergency stop button
- Status indicators

### AI Capabilities
- Natural language conversation
- Voice-activated commands
- Context-aware responses
- Ultra-fast response times (Groq)

### Computer Vision
- Face detection and tracking
- Real-time video streaming
- Human following mode
- Visual tracking indicators

## 🔐 Security Notes

**Important**: 
- Keep API keys private
- Don't commit `config.py` with real keys
- Use environment variables in production
- Secure your network connection

## 📄 License

MIT License - Feel free to use and modify!

## 👥 Credits

**Team Robomanthan**
- Advanced robotics and AI integration
- Computer vision and autonomous navigation
- Voice control and NLP capabilities

## 🚀 Future Enhancements

- [ ] Object detection (not just faces)
- [ ] Path planning and obstacle avoidance
- [ ] Multi-person tracking
- [ ] Gesture recognition
- [ ] Voice wake word detection
- [ ] Mobile app (native iOS/Android)
- [ ] Cloud integration
- [ ] Advanced AI personalities

## 📞 Support

For issues or questions:
1. Check documentation files
2. Run `python3 test_robot.py`
3. Review troubleshooting section
4. Open GitHub issue

---

**Built with ❤️ by Team Robomanthan**

🤖 Making robots smarter, one line of code at a time!
