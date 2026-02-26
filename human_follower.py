import serial
import time

class RobotBridge:
    def __init__(self, port='/dev/ttyUSB0', baud_rate=115200):
        self.ser = None
        
        try:
            # 1. Open Serial Connection
            # (Note: /dev/ttyACM0 is standard for Uno. If it fails, try /dev/ttyUSB0)
            self.ser = serial.Serial(port, baud_rate, timeout=1)
            
            # 2. Wait for Arduino to Reboot
            # When you connect Serial, Arduino Uno resets. We MUST wait.
            print(f"? Connecting to ESP32 on {port}...")
            time.sleep(2) 
            print("? Serial Bridge Established!")
            
        except Exception as e:
            print(f"?? Serial Connection Failed: {e}")
            print("? Check USB cable or try different port (ttyUSB0/ttyACM1)")
            self.ser = None

    def drive(self, left_speed, right_speed):
        """
        Sends command: "LEFT,RIGHT\n" (e.g., "100,-100\n")
        """
        if self.ser:
            try:
                # Format the string cleanly
                command = f"{int(left_speed)},{int(right_speed)}\n"
                
                # Send binary data
                self.ser.write(command.encode('utf-8'))
            except Exception as e:
                print(f"?? Serial Write Error: {e}")

    def stop(self):
        self.drive(0, 0)

    def close(self):
        if self.ser:
            self.stop()
            self.ser.close()
