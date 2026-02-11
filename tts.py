import pyttsx3
import threading
from servo_mouth import ServoMouth 

class RobotMouth:
    def __init__(self):
        # 1. Setup Audio Engine
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 1.0)
        except Exception as e:
            print(f"⚠️ TTS Init Error: {e}")

        # 2. Setup Servo Jaw
        self.jaw = ServoMouth(pin=12)

    def speak(self, text):
        print(f"🤖 Speaking: {text}")
        if not text: return
        
        # --- CALCULATE DURATION ---
        # A rough estimate is 0.5 seconds per word.
        # We ensure it runs for at least 2 seconds so short replies like "Hi" still move.
        word_count = len(text.split())
        duration = max(2.0, word_count * 0.5)

        # A. Start Servo Sweep in background thread
        # We pass the calculated 'duration' to the servo
        t = threading.Thread(target=self.jaw.move_mouth, args=(duration,))
        t.start()
        
        # B. Play Audio (Blocks code until done)
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"⚠️ TTS Error: {e}")
        
        # C. Wait for servo thread to finish if audio ended early
        t.join()
