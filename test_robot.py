#!/usr/bin/env python3
"""
Test script for Navis Robot components
Run this to verify everything works before running the full interface
"""

import sys
import time

print("=" * 60)
print("🧪 Navis Robot Component Test")
print("=" * 60)

# Test 1: Config
print("\n[1/7] Testing config.py...")
try:
    from config import (
        GROQ_API_KEY, SERIAL_PORT, SERIAL_BAUD, 
        MANUAL_SPEED, AUTO_SPEED, RASPBERRY_PI_IP
    )
    print(f"  ✅ Config loaded")
    print(f"     - Serial Port: {SERIAL_PORT}")
    print(f"     - Baud Rate: {SERIAL_BAUD}")
    print(f"     - Manual Speed: {MANUAL_SPEED}")
    print(f"     - Auto Speed: {AUTO_SPEED}")
    print(f"     - Raspberry Pi IP: {RASPBERRY_PI_IP}")
    if GROQ_API_KEY and GROQ_API_KEY != "YOUR_GROQ_API_KEY_HERE":
        print(f"     - Groq API: Configured ✅")
    else:
        print(f"     - Groq API: Not configured ⚠️")
except Exception as e:
    print(f"  ❌ Config error: {e}")
    sys.exit(1)

# Test 2: Serial Bridge
print("\n[2/7] Testing serial_bridge.py...")
try:
    from serial_bridge import RobotBridge
    print(f"  ✅ Serial bridge module loaded")
    
    # Try to connect (will fail if ESP32 not connected, but that's OK)
    print(f"  🔌 Attempting to connect to ESP32 on {SERIAL_PORT}...")
    bot = RobotBridge(port=SERIAL_PORT, baud_rate=SERIAL_BAUD)
    
    if bot.ser:
        print(f"  ✅ ESP32 connected!")
        print(f"  🧪 Testing motor commands...")
        
        # Test forward
        print(f"     - Forward (100, 100)...")
        bot.drive(100, 100)
        time.sleep(1)
        
        # Test backward
        print(f"     - Backward (-100, -100)...")
        bot.drive(-100, -100)
        time.sleep(1)
        
        # Test left
        print(f"     - Left (-100, 100)...")
        bot.drive(-100, 100)
        time.sleep(1)
        
        # Test right
        print(f"     - Right (100, -100)...")
        bot.drive(100, -100)
        time.sleep(1)
        
        # Stop
        print(f"     - Stop (0, 0)...")
        bot.stop()
        
        print(f"  ✅ Motor control working!")
        bot.close()
    else:
        print(f"  ⚠️  ESP32 not connected (this is OK if not plugged in)")
        print(f"     Check: USB cable, port name, ESP32 powered on")
        
except Exception as e:
    print(f"  ⚠️  Serial bridge error: {e}")
    print(f"     (This is expected if ESP32 is not connected)")

# Test 3: LLM Handler
print("\n[3/7] Testing llm_handler_updated.py...")
try:
    from llm_handler_updated import get_llm
    llm = get_llm()
    print(f"  ✅ LLM handler loaded")
    
    if GROQ_API_KEY and GROQ_API_KEY != "YOUR_GROQ_API_KEY_HERE":
        print(f"  🧪 Testing Groq API...")
        response = llm.ask("Say hello in 5 words or less")
        print(f"  ✅ Groq API working!")
        print(f"     Response: {response}")
    else:
        print(f"  ⚠️  Groq API key not configured")
        
except Exception as e:
    print(f"  ❌ LLM error: {e}")

# Test 4: TTS
print("\n[4/7] Testing tts.py...")
try:
    from tts import RobotMouth
    mouth = RobotMouth()
    print(f"  ✅ TTS module loaded")
    print(f"  🔊 Testing speech (you should hear this)...")
    mouth.speak("Testing robot speech")
    print(f"  ✅ TTS working!")
except Exception as e:
    print(f"  ⚠️  TTS error: {e}")
    print(f"     (This is expected if no audio output available)")

# Test 5: Camera
print("\n[5/7] Testing camera...")
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"  ✅ Camera working!")
            print(f"     Resolution: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print(f"  ⚠️  Camera opened but can't read frames")
        cap.release()
    else:
        print(f"  ⚠️  Camera not available")
        print(f"     (This is OK if no camera connected)")
except Exception as e:
    print(f"  ⚠️  Camera error: {e}")

# Test 6: Flask
print("\n[6/7] Testing Flask...")
try:
    from flask import Flask
    app = Flask(__name__)
    print(f"  ✅ Flask loaded")
except Exception as e:
    print(f"  ❌ Flask error: {e}")

# Test 7: Speech Recognition
print("\n[7/7] Testing speech recognition...")
try:
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    print(f"  ✅ Speech recognition loaded")
except Exception as e:
    print(f"  ❌ Speech recognition error: {e}")

# Summary
print("\n" + "=" * 60)
print("📊 Test Summary")
print("=" * 60)
print("\n✅ = Working")
print("⚠️  = Not available (but OK)")
print("❌ = Error (needs fixing)")

print("\n🎯 Next Steps:")
print("1. If ESP32 not connected, connect it via USB")
print("2. Upload esp32_motor_controller.ino to ESP32")
print("3. Run: python3 navis_complete_control.py")
print("4. Access: http://{}:5000".format(RASPBERRY_PI_IP))

print("\n" + "=" * 60)
