"""
Navis Robot - Main Conversational AI
Updated to use Groq/Perplexity instead of Gemini
"""

import sys
import os
import time
import socket
import datetime
from stt import RobotEar
from tts import RobotMouth
from llm_handler import get_llm
from config import ROBOT_NAME, ROBOT_CREATOR, VOSK_MODEL_PATH

# --- INITIALIZATION ---
try:
    print("🔌 Initializing Navis Systems...")
    
    # 1. Setup Modules
    mouth = RobotMouth()
    ear = RobotEar(model_path=VOSK_MODEL_PATH)  # Offline Vosk
    
    # 2. Setup LLM (Groq/Perplexity)
    llm = get_llm()
    
    print(f"✅ System Ready!")

except Exception as e:
    print(f"❌ Critical Error: {e}")
    sys.exit(1)

# --- HELPER FUNCTIONS ---

def check_internet():
    """Checks if the robot can reach the internet."""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

def get_time_greeting():
    hour = datetime.datetime.now().hour
    if hour < 12: 
        return "Good Morning"
    elif 12 <= hour < 18: 
        return "Good Afternoon"
    else: 
        return "Good Evening"

def check_reflex(text):
    """Fast, hardcoded answers for basic questions."""
    text = text.lower()

    if "who are you" in text or "your name" in text:
        return f"I am {ROBOT_NAME}, a humanoid robot built by {ROBOT_CREATOR}."
    
    if "built" in text or "creator" in text:
        return f"I was built by {ROBOT_CREATOR}, using Vosk for hearing and advanced AI for thinking."

    if "technical" in text or "specs" in text:
        return "I run on a Raspberry Pi 4B with offline hearing and cloud-based AI."

    if any(x in text for x in ["hi", "hello", "hey", "namaste"]):
        return f"{get_time_greeting()}! I am {ROBOT_NAME}."
        
    if any(x in text for x in ["stop", "exit", "quit", "power down", "shutdown"]):
        return "STOP_SYSTEM"

    return None

# --- BRAIN FUNCTION ---
def ask_ai_brain(text):
    """
    Sends queries to LLM (Groq/Perplexity).
    Automatically detects if internet access is needed.
    """
    if not check_internet():
        return "I cannot connect to the internet right now."

    print("🧠 Thinking...")
    
    try:
        # LLM handler automatically detects if question needs internet
        response = llm.ask(text, use_internet=llm.check_internet_keywords(text))
        return response
        
    except Exception as e:
        print(f"❌ AI Error: {e}")
        return "My AI brain is currently unreachable."

# --- MAIN LOOP ---
def main():
    # 1. Connection Check on Startup
    if check_internet():
        mouth.speak(f"{ROBOT_NAME} Online. AI Connected. {get_time_greeting()}.")
    else:
        mouth.speak(f"{ROBOT_NAME} Online. Warning: Internet Offline.")

    print(f"\n{'='*50}")
    print(f"🤖 {ROBOT_NAME} is listening...")
    print(f"{'='*50}\n")

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
                mouth.speak("Powering down. Goodbye!")
                break
            
            elif reflex_response:
                print(f"🤖 {ROBOT_NAME}: {reflex_response}")
                mouth.speak(reflex_response)
            
            # 4. ASK AI (Cloud)
            else:
                ai_response = ask_ai_brain(user_text)
                print(f"🤖 {ROBOT_NAME}: {ai_response}")
                mouth.speak(ai_response)

        except KeyboardInterrupt:
            print("\n🛑 Manual Interrupt.")
            break
        except Exception as e:
            print(f"⚠️ Loop Error: {e}")

    # Cleanup
    ear.close()
    print(f"👋 {ROBOT_NAME} Offline")

if __name__ == "__main__":
    main()
