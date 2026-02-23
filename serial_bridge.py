import serial
import time

class RobotBridge:
    def __init__(self, port='/dev/ttyUSB0', baud_rate=115200):
        self.ser = None
        self.port = port
        self.baud = baud_rate
        self.connect()

    def connect(self):
        try:
            # 1. Open Serial
            self.ser = serial.Serial(self.port, self.baud, timeout=0.1) # Low timeout for speed
            time.sleep(2) # Allow ESP32 to reset
            
            # 2. Flush any old garbage data
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            
            print(f"✅ Bridge Connected: {self.port}")
        except Exception as e:
            print(f"⚠️ Serial Error: {e}")
            self.ser = None

    def drive(self, left, right):
        """
        Sends command: "LEFT,RIGHT\n"
        """
        if self.ser:
            try:
                # Format: "150,150\n"
                command = f"{int(left)},{int(right)}\n"
                self.ser.write(command.encode('utf-8'))
                self.ser.flush() # FORCE send immediately (No lag)
            except:
                # If write fails, try to reconnect once
                print("⚠️ Connection Lost... Reconnecting")
                self.connect()

    def stop(self):
        self.drive(0, 0)
    
    def send_command(self, cmd):
        """Send raw command to Arduino"""
        if self.ser:
            try:
                self.ser.write(f"{cmd}\n".encode('utf-8'))
                self.ser.flush()
            except:
                print(f"⚠️ Command send failed: {cmd}")
    
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
        if self.ser:
            self.stop()
            self.ser.close()
