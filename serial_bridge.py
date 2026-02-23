import serial
import time

class RobotBridge:
    def __init__(self, esp32_port='/dev/ttyUSB0', mega_port='/dev/ttyUSB1', baud_rate=115200):
        self.esp32_ser = None
        self.mega_ser = None
        self.esp32_port = esp32_port
        self.mega_port = mega_port
        self.baud = baud_rate
        self.connect()

    def connect_serial(self, port_name, dev_path):
        """Attempt to connect a singular serial port"""
        try:
            ser = serial.Serial(dev_path, self.baud, timeout=0.1)
            time.sleep(2)
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            print(f"✅ {port_name} Connected: {dev_path}")
            return ser
        except Exception as e:
            print(f"⚠️ {port_name} Connection Error: {e}")
            return None

    def connect(self):
        """Connect both microcontrollers"""
        self.esp32_ser = self.connect_serial("ESP32 (Drive)", self.esp32_port)
        self.mega_ser = self.connect_serial("Arduino Mega (Hands)", self.mega_port)

    def drive(self, left, right):
        """
        Sends command: "LEFT,RIGHT\n"
        """
        if self.esp32_ser:
            try:
                # Format: "150,150\n"
                command = f"{int(left)},{int(right)}\n"
                self.esp32_ser.write(command.encode('utf-8'))
                self.esp32_ser.flush() # FORCE send immediately (No lag)
            except:
                # If write fails, try to reconnect once
                print("⚠️ ESP32 Connection Lost... Reconnecting")
                self.esp32_ser = self.connect_serial("ESP32 (Drive)", self.esp32_port)

    def stop(self):
        self.drive(0, 0)
    
    def send_command(self, cmd):
        """Send raw command to Arduino Mega (Hands/Arms)"""
        if self.mega_ser:
            try:
                self.mega_ser.write(f"{cmd}\n".encode('utf-8'))
                self.mega_ser.flush()
            except:
                print(f"⚠️ Arduino Mega Connection Lost... Reconnecting")
                self.mega_ser = self.connect_serial("Arduino Mega (Hands)", self.mega_port)
    
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
