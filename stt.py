import os
import sys
import json
import vosk
import pyaudio

class RobotEar:
    def __init__(self, model_path="model"):
        self.p = pyaudio.PyAudio()
        
        # Load Vosk Model (Offline) as per Doc [cite: 54-60]
        if not os.path.exists(model_path):
            print(f"❌ Error: Vosk model not found at '{model_path}'")
            print("Please download it: https://alphacephei.com/vosk/models")
            sys.exit(1)
            
        print(f"📥 Loading Vosk Model from {model_path}...")
        self.model = vosk.Model(model_path)
        self.rec = vosk.KaldiRecognizer(self.model, 16000)
        self.stream = None

    def listen(self):
        """
        Listens continuously using PyAudio and processes with Vosk.
        Returns text string when a full sentence is recognized.
        """
        # Open Stream (16kHz, Mono) [cite: 71-76]
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16, 
                channels=1, 
                rate=16000, 
                input=True, 
                frames_per_buffer=8000,
                input_device_index=None # Uses system default (Bluetooth if set)
            )
            
            print("🎤 Listening (Vosk Offline)...")
            
            while True:
                data = self.stream.read(4000, exception_on_overflow=False)
                
                # Process audio stream [cite: 86-87]
                if self.rec.AcceptWaveform(data):
                    result = json.loads(self.rec.Result())
                    text = result.get('text', '').strip()
                    
                    if text:
                        return text  # Return the recognized sentence

        except Exception as e:
            print(f"⚠️ Mic Error: {e}")
            return None

    def close(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
