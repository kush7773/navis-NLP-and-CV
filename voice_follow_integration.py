"""
Voice-Activated Follow Mode Integration
Combines voice commands with existing CV face tracking
"""

from flask import Flask, render_template, request, jsonify
from llm_handler_updated import get_llm
from tts import RobotMouth
import speech_recognition as sr
import io
import os
from pydub import AudioSegment
from config import (
    VOICE_PORT, 
    ROBOT_NAME, 
    FOLLOW_COMMANDS, 
    STOP_COMMANDS,
    RASPBERRY_PI_IP
)
import subprocess
import signal

app = Flask(__name__)

# Initialize components
try:
    llm = get_llm()
    mouth = RobotMouth()
    print(f"✅ Voice + Follow Integration Initialized")
except Exception as e:
    print(f"⚠️ Warning: Some components failed to initialize: {e}")
    llm = None
    mouth = None

# Global state for follow mode
follow_mode_active = False
navis_hybrid_process = None

def start_follow_mode():
    """Start the face tracking/follow mode (navis_hybrid.py)"""
    global navis_hybrid_process, follow_mode_active
    
    if navis_hybrid_process is None:
        try:
            # Start navis_hybrid.py in background
            navis_hybrid_process = subprocess.Popen(
                ['python3', 'navis_hybrid.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            follow_mode_active = True
            print("✅ Follow mode activated - navis_hybrid.py started")
            return True
        except Exception as e:
            print(f"❌ Failed to start follow mode: {e}")
            return False
    else:
        # Already running, just activate tracking
        follow_mode_active = True
        print("✅ Follow mode activated - tracking enabled")
        return True

def stop_follow_mode():
    """Stop the follow mode"""
    global navis_hybrid_process, follow_mode_active
    
    follow_mode_active = False
    
    # Note: We don't kill navis_hybrid.py, just disable tracking
    # The web interface at port 5000 stays active for manual control
    print("⏸️ Follow mode deactivated - tracking disabled")
    return True

def check_follow_command(text):
    """Check if text contains follow or stop command"""
    text_lower = text.lower()
    
    # Check for follow commands
    for cmd in FOLLOW_COMMANDS:
        if cmd in text_lower:
            return "FOLLOW"
    
    # Check for stop commands
    for cmd in STOP_COMMANDS:
        if cmd in text_lower:
            return "STOP"
    
    return None

@app.route('/')
def index():
    """Main voice control page"""
    return render_template('voice_follow_control.html')

@app.route('/audio_upload', methods=['POST'])
def audio_upload():
    """
    Receives audio from phone browser
    Converts to text, checks for follow/stop commands, or gets AI response
    """
    audio_file = request.files.get('audio')
    
    if not audio_file:
        return jsonify({'error': 'No audio received'}), 400
    
    try:
        # Save temporarily
        temp_webm = '/tmp/user_audio.webm'
        temp_wav = '/tmp/user_audio.wav'
        audio_file.save(temp_webm)
        
        # Convert webm to wav
        try:
            audio = AudioSegment.from_file(temp_webm, format="webm")
            audio.export(temp_wav, format="wav")
        except Exception as e:
            print(f"⚠️ Audio conversion error: {e}")
            temp_wav = temp_webm
        
        # Convert speech to text
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(temp_wav) as source:
            audio_data = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio_data)
            print(f"👤 User (via web): {text}")
        except sr.UnknownValueError:
            return jsonify({'error': 'Could not understand audio'}), 400
        except sr.RequestError as e:
            print(f"⚠️ Google SR error: {e}")
            return jsonify({'error': 'Speech recognition service unavailable'}), 500
        
        # Check for follow/stop commands FIRST
        command = check_follow_command(text)
        
        if command == "FOLLOW":
            success = start_follow_mode()
            if success:
                response = f"Follow mode activated! I will track and follow you using my camera."
            else:
                response = "Sorry, I couldn't activate follow mode."
            
            print(f"🤖 {ROBOT_NAME}: {response}")
            if mouth:
                mouth.speak(response)
            
            return jsonify({
                'transcription': text,
                'response': response,
                'command': 'follow',
                'follow_active': follow_mode_active
            })
        
        elif command == "STOP":
            stop_follow_mode()
            response = "Stopping. Follow mode deactivated."
            
            print(f"🤖 {ROBOT_NAME}: {response}")
            if mouth:
                mouth.speak(response)
            
            return jsonify({
                'transcription': text,
                'response': response,
                'command': 'stop',
                'follow_active': follow_mode_active
            })
        
        # Not a command, get AI response
        else:
            if llm:
                response = llm.ask(text)
            else:
                response = "AI is not available right now."
            
            print(f"🤖 {ROBOT_NAME}: {response}")
            
            if mouth:
                mouth.speak(response)
            
            return jsonify({
                'transcription': text,
                'response': response,
                'follow_active': follow_mode_active
            })
        
        # Clean up temp files
        try:
            if os.path.exists(temp_webm):
                os.remove(temp_webm)
            if os.path.exists(temp_wav) and temp_wav != temp_webm:
                os.remove(temp_wav)
        except:
            pass
        
    except Exception as e:
        print(f"❌ Audio processing error: {e}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/text_chat', methods=['POST'])
def text_chat():
    """
    Text-based chat endpoint with follow command support
    """
    data = request.json
    user_text = data.get('message', '').strip()
    
    if not user_text:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        print(f"👤 User (via text): {user_text}")
        
        # Check for follow/stop commands
        command = check_follow_command(user_text)
        
        if command == "FOLLOW":
            success = start_follow_mode()
            response = "Follow mode activated! I will track and follow you." if success else "Sorry, couldn't activate follow mode."
            
        elif command == "STOP":
            stop_follow_mode()
            response = "Stopping. Follow mode deactivated."
            
        else:
            # Get AI response
            if llm:
                response = llm.ask(user_text)
            else:
                response = "AI is not available right now."
        
        print(f"🤖 {ROBOT_NAME}: {response}")
        
        if mouth:
            mouth.speak(response)
        
        return jsonify({
            'response': response,
            'follow_active': follow_mode_active
        })
        
    except Exception as e:
        print(f"❌ Text chat error: {e}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/follow_status')
def follow_status():
    """Get current follow mode status"""
    return jsonify({
        'follow_active': follow_mode_active,
        'hybrid_running': navis_hybrid_process is not None
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'robot': ROBOT_NAME,
        'llm_available': llm is not None,
        'tts_available': mouth is not None,
        'follow_active': follow_mode_active,
        'ip': RASPBERRY_PI_IP
    })

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"🌐 {ROBOT_NAME} Voice + Follow Control Starting...")
    print(f"{'='*60}")
    print(f"\n📱 Access from your phone:")
    print(f"   http://{RASPBERRY_PI_IP}:{VOICE_PORT}")
    print(f"\n🎤 Voice Commands:")
    print(f"   - 'Follow me Navis' → Activates face tracking")
    print(f"   - 'Stop Navis' → Stops following")
    print(f"   - Any other question → AI response")
    print(f"\n📹 Vision Control (if needed):")
    print(f"   http://{RASPBERRY_PI_IP}:5000")
    print(f"\n{'='*60}\n")
    
    try:
        app.run(host='0.0.0.0', port=VOICE_PORT, debug=False, threaded=True)
    finally:
        # Cleanup on exit
        if navis_hybrid_process:
            navis_hybrid_process.terminate()
