import serial
import time

class RobotBridge:
    def __init__(self, esp32_port='/dev/ttyESP32', mega_port='/dev/ttyArduino', baud_rate=115200):
        self.esp32_ser = None
        self.mega_ser = None
        
        # ESP32 — Drive Motors
        try:
            self.esp32_ser = serial.Serial(esp32_port, baud_rate, timeout=1)
            print(f"⚡ Connecting to ESP32 on {esp32_port}...")
            time.sleep(2)  # Wait for Arduino/ESP32 reboot on serial connect
            print("✅ ESP32 (Drive) Connected!")
        except Exception as e:
            print(f"⚠️ ESP32 Connection Failed: {e}")
            self.esp32_ser = None
        
        # Arduino Mega — Hands/Arms
        try:
            self.mega_ser = serial.Serial(mega_port, baud_rate, timeout=1)
            print(f"⚡ Connecting to Arduino Mega on {mega_port}...")
            time.sleep(2)
            print("✅ Arduino Mega (Hands) Connected!")
        except Exception as e:
            print(f"⚠️ Arduino Mega Connection Failed: {e}")
            self.mega_ser = None

    def drive(self, left_speed, right_speed):
        """
        Sends command: "LEFT,RIGHT\n" (e.g., "150,-150\n")
        Non-blocking write — NO flush() to avoid stalling the frame loop.
        """
        if self.esp32_ser:
            try:
                command = f"{int(left_speed)},{int(right_speed)}\n"
                self.esp32_ser.write(command.encode('utf-8'))
            except Exception as e:
                print(f"⚠️ ESP32 Write Error: {e}")

    def stop(self):
        self.drive(0, 0)
    
    def send_command(self, cmd):
        """Send raw command to Arduino Mega (Hands/Arms)"""
        if self.mega_ser:
            try:
                self.mega_ser.write(f"{cmd}\n".encode('utf-8'))
            except Exception as e:
                print(f"⚠️ Arduino Mega Write Error: {e}")
    
    # ============================================
    #   HAND CONTROLS
    # ============================================
    def open_left_hand(self):
        """Open left hand (all fingers)"""
        self.send_command("LO")
    
    def close_left_hand(self):
        """Close left hand (all fingers)"""
        self.send_command("LC")
    
    def open_right_hand(self):
        """Open right hand (all fingers)"""
        self.send_command("RO")
    
    def close_right_hand(self):
        """Close right hand (all fingers)"""
        self.send_command("RC")
    
    # ============================================
    #   WRIST CONTROLS
    # ============================================
    def set_left_wrist(self, angle):
        """Set left wrist angle (0-180)"""
        angle = max(0, min(180, int(angle)))
        self.send_command(f"WL{angle}")
    
    def set_right_wrist(self, angle):
        """Set right wrist angle (0-180)"""
        angle = max(0, min(180, int(angle)))
        self.send_command(f"WR{angle}")
    
    # ============================================
    #   BICEP CONTROLS
    # ============================================
    def left_bicep_up(self):
        """Move left bicep to up position"""
        self.send_command("LU")
    
    def left_bicep_down(self):
        """Move left bicep to down position"""
        self.send_command("LD")
    
    def right_bicep_up(self):
        """Move right bicep to up position"""
        self.send_command("RU")
    
    def right_bicep_down(self):
        """Move right bicep to down position"""
        self.send_command("RD")
    
    def stop_biceps(self):
        """Emergency stop for all bicep motors"""
        self.send_command("BS")

    def close(self):
        self.stop()
        if self.esp32_ser:
            self.esp32_ser.close()
            self.esp32_ser = None
        if self.mega_ser:
            self.mega_ser.close()
            self.mega_ser = None
