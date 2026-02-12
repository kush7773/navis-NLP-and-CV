import serial.tools.list_ports
import time

print("🔍 Scanning for serial ports...")
try:
    ports = list(serial.tools.list_ports.comports())

    if not ports:
        print("❌ No serial ports found! Check USB cables and power.")
    else:
        for port in ports:
            print(f"✅ Found Device: {port.device}")
            print(f"   Description: {port.description}")
            print(f"   Hardware ID: {port.hwid}")
            print("-" * 30)
            
    print("\n💡 NOTE: Arduino Mega often appears as '/dev/ttyACM0'. ESP32 usually as '/dev/ttyUSB0'.")
except Exception as e:
    print(f"❌ Error scanning ports: {e}")
