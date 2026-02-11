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
            
            print(f"? Bridge Connected: {self.port}")
        except Exception as e:
            print(f"?? Serial Error: {e}")
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
                print("?? Connection Lost... Reconnecting")
                self.connect()

    def stop(self):
        self.drive(0, 0)

    def close(self):
        if self.ser:
            self.stop()
            self.ser.close()
