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
            
            # Find and set a male voice
            voices = self.engine.getProperty('voices')
            male_voice_found = False
            
            for voice in voices:
                voice_id_lower = voice.id.lower()
                voice_name_lower = voice.name.lower()
                # Check gender metadata or common male names/identifiers
                if (hasattr(voice, 'gender') and voice.gender == 'male') or \
                   'male' in voice_name_lower or \
                   'david' in voice_id_lower or \
                   'alex' in voice_id_lower:
                    self.engine.setProperty('voice', voice.id)
                    male_voice_found = True
                    print(f"✅ TTS selected male voice: {voice.name}")
                    break
            
            if not male_voice_found and len(voices) > 0:
                # Default to the first voice typically male in SAPI5/espeak
                self.engine.setProperty('voice', voices[0].id)
                print(f"⚠️ Male voice not explicitly found, defaulting to: {voices[0].name}")
                
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
