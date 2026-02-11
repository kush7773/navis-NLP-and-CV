import sys
import os
import time
import socket
import datetime
import google.generativeai as genai
from stt import RobotEar
from tts import RobotMouth

# --- CONFIGURATION ---
# Your specific key
GEMINI_API_KEY = "AIzaSyAjMIaj5XprC8EZdZJU6JdMhynDbvcSiGk" 

# --- INITIALIZATION ---
try:
    print("🔌 Initializing Navis Systems...")
    
    # 1. Setup Modules
    mouth = RobotMouth()
    ear = RobotEar(model_path="model") # Offline Vosk
    
    # 2. Setup Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    
    # USING GEMINI 2.5 FLASH LITE AS REQUESTED
    MODEL_NAME = 'gemini-2.5-flash-lite'
    model = genai.GenerativeModel(MODEL_NAME)
    
    print(f"✅ System Ready. Brain: {MODEL_NAME}")

except Exception as e:
    print(f"❌ Critical Error: {e}")
    sys.exit(1)

# --- HELPER FUNCTIONS ---

def check_internet():
    """Checks if the robot can reach Google's servers."""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

def get_time_greeting():
    hour = datetime.datetime.now().hour
    if hour < 12: return "Good Morning"
    elif 12 <= hour < 18: return "Good Afternoon"
    else: return "Good Evening"

def check_reflex(text):
    """Fast, hardcoded answers for basic questions."""
    text = text.lower()

    if "who are you" in text or "your name" in text:
        return "I am Navis, a humanoid robot built by Team Robomanthan."
    
    if "built" in text or "creator" in text:
        return "I was built by Team Robomanthan, using Vosk for hearing and Gemini for thinking."

    if "technical" in text or "specs" in text:
        return "I run on a Raspberry Pi 4B with offline hearing and Gemini 2.5 AI."

    if any(x in text for x in ["hi", "hello", "hey", "namaste"]):
        return f"{get_time_greeting()}! I am Navis."
        
    if any(x in text for x in ["stop", "exit", "quit", "power down"]):
        return "STOP_SYSTEM"

    return None

# --- BRAIN FUNCTION ---
def ask_gemini(text):
    """
    Sends queries to Gemini. 
    Includes a 'Safety Fallback' to prevent crashes if Tools fail.
    """
    if not check_internet():
        return "I cannot connect to the internet right now."

    print("🧠 Thinking (Gemini)...")
    
    prompt = f"""You are Navis, a friendly robot built by Team Robomanthan.
    User asks: "{text}"
    Answer briefly (max 2 sentences)."""

    try:
        # ATTEMPT 1: Google Search Tool
        # We try the standard tool definition
        tools = {'google_search': {}}
        response = model.generate_content(prompt, tools=tools)
        return response.text.strip()
        
    except Exception as e:
        print(f"⚠️ Search Tool Failed (Switching to Standard Brain): {e}")
        try:
            # ATTEMPT 2: Fallback to Standard Brain (No Tools)
            # This bypasses the "Unknown field" error so the robot still talks
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e2:
            print(f"❌ Gemini Error: {e2}")
            return "My AI brain is currently unreachable."

# --- MAIN LOOP ---
def main():
    # 1. Connection Check on Startup
    if check_internet():
        mouth.speak(f"Navis Online. Connected to Gemini 2 point 5. {get_time_greeting()}.")
    else:
        mouth.speak("Navis Online. Warning: Internet Offline.")

    while True:
        try:
            # 2. LISTEN (Vosk Offline)
            user_text = ear.listen()
            
            if not user_text:
                continue
            
            print(f"\n👤 You: {user_text}")

            # 3. CHECK REFLEXES (Local)
            reflex_response = check_reflex(user_text)
            
            if reflex_response == "STOP_SYSTEM":
                mouth.speak("Powering down.")
                break
            
            elif reflex_response:
                mouth.speak(reflex_response)
            
            # 4. ASK GEMINI (Cloud)
            else:
                ai_response = ask_gemini(user_text)
                mouth.speak(ai_response)

        except KeyboardInterrupt:
            print("\n🛑 Manual Interrupt.")
            break
        except Exception as e:
            print(f"⚠️ Loop Error: {e}")

    # Cleanup
    ear.close()
    print("👋 System Offline")

if __name__ == "__main__":
    main()
