# 🚀 Quick Start Guide - Navis Robot

## ✅ Your Configuration

- **Groq API**: ✅ Configured
- **Raspberry Pi IP**: `192.168.0.182`
- **Free Alternative**: Hugging Face (optional)

---

## 🎯 Run Your Robot (3 Steps)

### Step 1: Install Dependencies (One Time)

```bash
cd /Users/tokenadmin/Desktop/python
chmod +x setup_navis.sh
./setup_navis.sh
```

This installs everything automatically!

---

### Step 2: Start the Robot

```bash
python3 voice_follow_integration.py
```

You'll see:
```
🌐 Navis Voice + Follow Control Starting...
📱 Access from your phone:
   http://192.168.0.182:5000
```

---

### Step 3: Control from Phone

1. **Connect phone to same WiFi** as Raspberry Pi
2. **Open browser**: `http://192.168.0.182:5000`
3. **Allow microphone** when prompted
4. **Start talking!**

---

## 🎤 Voice Commands

### Follow Mode
- **"Follow me Navis"** → Robot starts tracking and following you with camera
- **"Stop Navis"** → Robot stops following

### Normal Conversation
- **"Who are you?"** → AI responds
- **"Tell me a joke"** → AI tells a joke
- **Any question** → AI answers

---

## 🎮 Quick Command Buttons

On the web interface, you'll see two buttons:
- **🎯 Follow Me** → Instant follow mode activation
- **⏹️ Stop** → Instant stop

---

## 📹 How Follow Mode Works

1. You say **"Follow me Navis"**
2. Robot activates camera face tracking (from `navis_hybrid.py`)
3. Robot identifies and locks onto your face
4. Robot follows you using motors
5. You say **"Stop Navis"** to stop

The robot uses your existing CV code for face detection and tracking!

---

## 🆓 Free LLM Options

### Currently Configured: Groq ✅
- **Speed**: Ultra-fast ⚡⚡⚡
- **Cost**: FREE (unlimited for now)
- **Your API Key**: Already configured!

### Optional: Hugging Face
If you want an alternative:

1. Get free API key: https://huggingface.co/settings/tokens
2. Edit `config.py`:
   ```python
   HUGGINGFACE_API_KEY = "hf_your_key_here"
   PRIMARY_LLM = "huggingface"  # Switch to HF
   ```

---

## 🔧 Files Overview

### Main Files (Use These)

| File | Purpose | Run Command |
|------|---------|-------------|
| `voice_follow_integration.py` | **Main program** - Voice + Follow | `python3 voice_follow_integration.py` |
| `config.py` | Settings (already configured!) | Edit if needed |
| `llm_handler_updated.py` | Groq + HuggingFace LLM | Auto-loaded |

### Supporting Files (Already Working)

| File | Purpose |
|------|---------|
| `tts.py` | Speech output + servo mouth |
| `stt.py` | Offline speech recognition |
| `servo_mouth.py` | Mouth movement |
| `navis_hybrid.py` | Camera face tracking |
| `serial_bridge.py` | Motor control |

---

## 📱 Web Interface Features

### Voice Control
- Hold button to record
- Release to send
- Robot speaks response
- Servo mouth moves automatically!

### Follow Mode Indicator
- **Green + Glowing**: Follow mode ACTIVE
- **Red**: Follow mode inactive

### Text Input
- Type instead of speak
- Same commands work
- Great for quiet environments

---

## 🎬 Example Session

```
You: [Hold button] "Follow me Navis"
Robot: "Follow mode activated! I will track and follow you using my camera."
[Robot's camera activates, finds your face, starts following]

You: [Walking around]
Robot: [Follows you, keeping you in center of camera view]

You: [Hold button] "Stop Navis"
Robot: "Stopping. Follow mode deactivated."
[Robot stops moving]

You: [Hold button] "Who are you?"
Robot: "I am Navis, a friendly humanoid robot built by Team Robomanthan."
```

---

## 🐛 Troubleshooting

### Can't access web interface
```bash
# Check Raspberry Pi IP
hostname -I

# Should show: 192.168.0.182
# If different, update config.py
```

### "Groq API error"
- Check internet connection: `ping google.com`
- API key is already configured, should work!

### Follow mode doesn't start
- Make sure camera is connected
- Check `navis_hybrid.py` works: `python3 navis_hybrid.py`

### Servo not moving
- Check GPIO pin 12 connection
- Verify servo power supply

---

## 🎯 What's Different from Before

### ❌ Removed
- Google Gemini (you wanted free alternative)
- Perplexity (paid service)

### ✅ Added
- **Groq API** (ultra-fast, free)
- **Hugging Face** (free alternative option)
- **Voice-activated follow mode**
- **Quick command buttons**
- **Follow mode indicator**

### ✅ Kept (Your Existing Features)
- Servo mouth movement
- Camera face tracking
- Motor control
- Manual web controls
- Offline speech recognition

---

## 🚀 Ready to Go!

Everything is configured with your:
- ✅ Groq API key
- ✅ Raspberry Pi IP (192.168.0.182)
- ✅ Follow mode commands
- ✅ Free LLM options

Just run:
```bash
python3 voice_follow_integration.py
```

Then open on your phone:
```
http://192.168.0.182:5000
```

**Start talking to your robot!** 🤖✨
