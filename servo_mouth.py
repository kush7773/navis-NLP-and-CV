import RPi.GPIO as GPIO
import time

class ServoMouth:
    def __init__(self, pin=12):
        self.pin = pin
        self.pwm = None
        
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin, GPIO.OUT)
            
            # Setup PWM at 50Hz
            self.pwm = GPIO.PWM(pin, 50)
            self.pwm.start(0) 
            
            print(f"? Safe Servo Initialized on GPIO {pin} (REVERSED)")
            
            # REVERSED START: Start at 180 (Closed) instead of 0
            self._set_safe_angle(180)
            time.sleep(0.3)
            self.pwm.ChangeDutyCycle(0) 
            
        except Exception as e:
            print(f"?? Servo Init Error: {e}")
            self.pwm = None

    def _set_safe_angle(self, angle):
        if self.pwm:
            # Standard mapping: 0-180 -> 3-11 duty cycle
            duty = 3 + (angle / 180) * 8  
            self.pwm.ChangeDutyCycle(duty)

    def move_mouth(self, duration):
        """
        Reversed Animation: Moves DOWN from 180 to 140.
        """
        if not self.pwm:
            return

        start_time = time.time()
        end_time = start_time + duration
        
        try:
            while time.time() < end_time:
                # OPEN MOUTH (Move DOWN: 180 -> 140)
                # We start at 180 and subtract 5 until we hit 140
                for angle in range(180, 140, -5): 
                    self._set_safe_angle(angle)
                    time.sleep(0.01) 
                
                time.sleep(0.05)

                # CLOSE MOUTH (Move UP: 140 -> 180)
                # We start at 140 and add 5 until we hit 180
                for angle in range(140, 181, 5):
                    self._set_safe_angle(angle)
                    time.sleep(0.01)
                
                time.sleep(0.05)
                
        except Exception as e:
            print(f"Servo Glitch: {e}")
            
        finally:
            # REVERSED END: Ensure mouth is closed at 180
            self._set_safe_angle(180)
            time.sleep(0.2)
            self.pwm.ChangeDutyCycle(0) 

    def cleanup(self):
        if self.pwm:
            self.pwm.stop()
        GPIO.cleanup()
