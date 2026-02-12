import serial
import time

class RobotBridge:
    def __init__(self, wheel_port='/dev/ttyUSB0', hand_port='/dev/ttyUSB1', baud_rate=115200):
        self.ser_wheels = None
        self.ser_hands = None
        self.wheel_port = wheel_port
        self.hand_port = hand_port
        self.baud = baud_rate
        self.connect()

    def connect(self):
        # Connect Wheels (ESP32)
        try:
            self.ser_wheels = serial.Serial(self.wheel_port, self.baud, timeout=0.1)
            time.sleep(2)
            self.ser_wheels.reset_input_buffer()
            print(f"✅ Wheels Connected: {self.wheel_port}")
        except Exception as e:
            print(f"⚠️ Wheels Serial Error: {e}")
            self.ser_wheels = None

        # Connect Hands (Mega)
        try:
            self.ser_hands = serial.Serial(self.hand_port, self.baud, timeout=0.1)
            time.sleep(2)
            self.ser_hands.reset_input_buffer()
            print(f"✅ Hands Connected: {self.hand_port}")
        except Exception as e:
            print(f"⚠️ Hands Serial Error: {e}")
            self.ser_hands = None

    def drive(self, left, right):
        """Sends command to WHEELS: 'LEFT,RIGHT\n'"""
        if self.ser_wheels:
            try:
                command = f"{int(left)},{int(right)}\n"
                self.ser_wheels.write(command.encode('utf-8'))
                self.ser_wheels.flush()
            except:
                print("⚠️ Wheels Lost... Reconnecting")
                self.connect()

    def stop(self):
        self.drive(0, 0)
        # Also stop biceps?
        self.send_command("BS")
    
    def send_command(self, cmd):
        """Send command to HANDS (Arduino Mega)"""
        if self.ser_hands:
            try:
                self.ser_hands.write(f"{cmd}\n".encode('utf-8'))
                self.ser_hands.flush()
            except:
                print(f"⚠️ Hands Comm Failed: {cmd}")
        else:
            # Fallback: some users might have everything on one port? 
            # But here we STRICTLY separate.
            pass
    
    # ============================================
    #   HAND CONTROLS
    # ============================================
    def open_left_hand(self):
        """Open left hand (all fingers)"""
        self.send_command("LC")
    
    def close_left_hand(self):
        """Close left hand (all fingers)"""
        self.send_command("LO")
    
    def open_right_hand(self):
        """Open right hand (all fingers)"""
        self.send_command("RC")
    
    def close_right_hand(self):
        """Close right hand (all fingers)"""
        self.send_command("RO")
    
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
        self.send_command("BLU")
    
    def left_bicep_down(self):
        """Move left bicep to down position"""
        self.send_command("BLD")
    
    def right_bicep_up(self):
        """Move right bicep to up position"""
        self.send_command("BRU")
    
    def right_bicep_down(self):
        """Move right bicep to down position"""
        self.send_command("BRD")
    
    def stop_biceps(self):
        """Emergency stop for all bicep motors"""
        self.send_command("BS")

    def close(self):
        if self.ser:
            self.stop()
            self.ser.close()
