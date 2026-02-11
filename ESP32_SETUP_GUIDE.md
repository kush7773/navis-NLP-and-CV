# 🔧 ESP32 Motor Controller Setup Guide

## ✅ Hardware Configuration

### ESP32 Pin Connections

```
ESP32 Pin → Motor Driver
─────────────────────────
GPIO 32   → Left Motor RPWM (Forward)
GPIO 33   → Left Motor LPWM (Backward)
GPIO 25   → Right Motor RPWM (Forward)
GPIO 26   → Right Motor LPWM (Backward)
```

### Communication
- **Raspberry Pi (Master)** ↔ **ESP32 (Slave)**
- **Protocol**: Serial UART
- **Baud Rate**: 115200
- **Connection**: USB cable (Raspberry Pi → ESP32)

---

## 📁 Files Overview

### ESP32 Arduino Code
**File**: `esp32_motor_controller/esp32_motor_controller.ino`

**Features**:
- Serial communication at 115200 baud
- PWM motor control (20kHz, 8-bit resolution)
- Command format: `LEFT,RIGHT\n`
- Speed range: -255 to 255
- Automatic motor direction control

### Python Serial Bridge
**File**: `serial_bridge.py`

**Features**:
- Sends `LEFT,RIGHT\n` commands to ESP32
- Auto-reconnect on connection loss
- Low latency (0.1s timeout)
- Compatible with ESP32 protocol

---

## 🚀 Setup Steps

### Step 1: Upload to ESP32

1. **Open Arduino IDE**
2. **Install ESP32 Board**:
   - Go to: File → Preferences
   - Add URL: `https://dl.espressif.com/dl/package_esp32_index.json`
   - Go to: Tools → Board → Boards Manager
   - Search "ESP32" and install

3. **Select Board**:
   - Tools → Board → ESP32 Dev Module

4. **Open Code**:
   - File → Open → `esp32_motor_controller.ino`

5. **Upload**:
   - Connect ESP32 via USB
   - Tools → Port → Select your ESP32 port
   - Click Upload ⬆️

### Step 2: Verify ESP32 is Working

1. **Open Serial Monitor** (Tools → Serial Monitor)
2. **Set baud to 115200**
3. **You should see**:
   ```
   ESP32 Motor Controller Starting...
   ✅ ESP32 Ready!
   Waiting for commands from Raspberry Pi...
   Format: LEFT,RIGHT (e.g., 150,-100)
   ```

### Step 3: Test Commands Manually

In Serial Monitor, type:
```
150,150    → Both motors forward
-100,-100  → Both motors backward
150,-150   → Turn right
-150,150   → Turn left
0,0        → Stop
```

You should see:
```
✅ L:150 R:150
✅ L:-100 R:-100
```

---

## 🔌 Raspberry Pi Connection

### Find ESP32 Port

On Raspberry Pi, run:
```bash
ls /dev/ttyUSB*
# or
ls /dev/ttyACM*
```

Common ports:
- `/dev/ttyUSB0` (most common)
- `/dev/ttyACM0`
- `/dev/ttyUSB1`

### Update config.py

If your port is different, edit:
```python
SERIAL_PORT = "/dev/ttyUSB0"  # Change to your port
SERIAL_BAUD = 115200          # Must match ESP32
```

---

## 🎮 Motor Speed Settings

### ESP32 Constants (Already Set)
```cpp
#define BSPEED    55   // Base Speed
#define SPEED     85   // Normal Speed
#define TURN_SPD  100  // Turning Speed
```

### Python config.py (Already Updated)
```python
BASE_SPEED = 55      # Matches ESP32 BSPEED
AUTO_SPEED = 85      # Matches ESP32 SPEED
MANUAL_SPEED = 100   # Matches ESP32 TURN_SPD
TURN_SPEED = 100     # Matches ESP32 TURN_SPD
```

✅ **All speeds are synchronized!**

---

## 📡 Communication Protocol

### Command Format
```
LEFT,RIGHT\n
```

### Examples
| Command | Left Motor | Right Motor | Action |
|---------|------------|-------------|--------|
| `100,100\n` | 100 forward | 100 forward | Move forward |
| `-100,-100\n` | 100 backward | 100 backward | Move backward |
| `100,-100\n` | 100 forward | 100 backward | Turn right |
| `-100,100\n` | 100 backward | 100 forward | Turn left |
| `0,0\n` | Stop | Stop | Stop |

### Python Usage
```python
from serial_bridge import RobotBridge

bot = RobotBridge(port='/dev/ttyUSB0', baud_rate=115200)

# Forward
bot.drive(100, 100)

# Backward
bot.drive(-100, -100)

# Turn right
bot.drive(100, -100)

# Turn left
bot.drive(-100, 100)

# Stop
bot.stop()
```

---

## 🔧 Troubleshooting

### ESP32 Not Responding

**Check**:
1. USB cable connected?
2. ESP32 powered on?
3. Correct port in `config.py`?
4. Baud rate is 115200?

**Test**:
```bash
# On Raspberry Pi
python3 -c "from serial_bridge import RobotBridge; bot = RobotBridge(); bot.drive(100,100)"
```

### Motors Not Moving

**Check**:
1. Motor driver powered?
2. ESP32 pins connected correctly?
3. Motor driver enable pin high?
4. Battery/power supply sufficient?

**Test ESP32**:
Open Serial Monitor and send: `100,100`
You should see: `✅ L:100 R:100`

### Wrong Direction

If motors spin opposite direction:
- Swap motor wires on motor driver
- OR modify ESP32 code to invert speeds

---

## ✅ Verification Checklist

- [ ] ESP32 code uploaded successfully
- [ ] Serial Monitor shows "✅ ESP32 Ready!"
- [ ] Manual commands work in Serial Monitor
- [ ] Raspberry Pi can find ESP32 port
- [ ] `config.py` has correct port and baud rate
- [ ] Python can send commands to ESP32
- [ ] Motors respond to commands
- [ ] All directions work correctly (F/B/L/R)

---

## 🎯 Integration with Control Interface

The ESP32 is automatically used by:

1. **navis_complete_control.py** - Professional UI
   - Joystick controls (F/B/L/R/S)
   - Emergency stop
   - Follow mode

2. **navis_hybrid.py** - Camera interface
   - Face tracking
   - Manual controls

3. **voice_follow_integration.py** - Voice control
   - Voice-activated following

All use `serial_bridge.py` which communicates with ESP32!

---

## 📊 System Architecture

```
┌─────────────────────────────────────┐
│      Raspberry Pi (Master)          │
│                                     │
│  ┌───────────────────────────────┐ │
│  │  Python Control Programs      │ │
│  │  - navis_complete_control.py  │ │
│  │  - navis_hybrid.py            │ │
│  │  - voice_follow_integration.py│ │
│  └───────────────┬───────────────┘ │
│                  │                  │
│  ┌───────────────▼───────────────┐ │
│  │  serial_bridge.py             │ │
│  │  Sends: "LEFT,RIGHT\n"        │ │
│  └───────────────┬───────────────┘ │
└──────────────────┼──────────────────┘
                   │ USB Serial
                   │ 115200 baud
┌──────────────────▼──────────────────┐
│      ESP32 (Slave)                  │
│                                     │
│  ┌───────────────────────────────┐ │
│  │  esp32_motor_controller.ino   │ │
│  │  Receives: "LEFT,RIGHT\n"     │ │
│  │  Parses and controls motors   │ │
│  └───────────────┬───────────────┘ │
│                  │                  │
│  ┌───────────────▼───────────────┐ │
│  │  PWM Motor Control            │ │
│  │  GPIO 32,33,25,26             │ │
│  └───────────────┬───────────────┘ │
└──────────────────┼──────────────────┘
                   │
┌──────────────────▼──────────────────┐
│      Motor Driver                   │
│      (L298N / BTS7960 / etc)        │
└──────────────────┬──────────────────┘
                   │
┌──────────────────▼──────────────────┐
│      DC Motors                      │
│      Left Motor + Right Motor       │
└─────────────────────────────────────┘
```

---

## ✅ Summary

**ESP32 Configuration**: ✅ Complete
- Pins: L_FWD=32, L_BWD=33, R_FWD=25, R_BWD=26
- PWM: 20kHz, 8-bit (0-255)
- Serial: 115200 baud
- Protocol: `LEFT,RIGHT\n`

**Python Configuration**: ✅ Complete
- Serial bridge working
- Speed settings synchronized
- All control programs integrated

**Ready to use!** 🚀
