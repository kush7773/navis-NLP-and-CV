"""
Web Voice Interface for Navis Robot
Allows phone to be used as microphone via web browser
"""

from flask import Flask, render_template, request, jsonify
from llm_handler import get_llm
from tts import RobotMouth
import speech_recognition as sr
import io
import os
from pydub import AudioSegment
from config import VOICE_PORT, ROBOT_NAME

app = Flask(__name__)

# Initialize components
try:
    llm = get_llm()
    mouth = RobotMouth()
    print(f"✅ Web Voice Interface Initialized")
except Exception as e:
    print(f"⚠️ Warning: Some components failed to initialize: {e}")
    llm = None
    mouth = None

@app.route('/')
def index():
    """Main voice control page"""
    return render_template('voice_control.html')

@app.route('/audio_upload', methods=['POST'])
def audio_upload():
    """
    Receives audio from phone browser
    Converts to text, gets AI response, speaks it
    """
    audio_file = request.files.get('audio')
    
    if not audio_file:
        return jsonify({'error': 'No audio received'}), 400
    
    try:
        # Save temporarily
        temp_webm = '/tmp/user_audio.webm'
        temp_wav = '/tmp/user_audio.wav'
        audio_file.save(temp_webm)
        
        # Convert webm to wav (speech_recognition needs wav)
        try:
            audio = AudioSegment.from_file(temp_webm, format="webm")
            audio.export(temp_wav, format="wav")
        except Exception as e:
            print(f"⚠️ Audio conversion error: {e}")
            # Try using the file directly
            temp_wav = temp_webm
        
        # Convert speech to text
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(temp_wav) as source:
            audio_data = recognizer.record(source)
        
        # Try Google Speech Recognition (free, cloud-based)
        try:
            text = recognizer.recognize_google(audio_data)
            print(f"👤 User (via web): {text}")
        except sr.UnknownValueError:
            return jsonify({'error': 'Could not understand audio'}), 400
        except sr.RequestError as e:
            print(f"⚠️ Google SR error: {e}")
            return jsonify({'error': 'Speech recognition service unavailable'}), 500
        
        # Get AI response
        if llm:
            # Check if question needs internet
            needs_internet = llm.check_internet_keywords(text)
            response = llm.ask(text, use_internet=needs_internet)
        else:
            response = "AI is not available right now."
        
        print(f"🤖 {ROBOT_NAME}: {response}")
        
        # Speak response (with servo mouth movement)
        if mouth:
            mouth.speak(response)
        
        # Clean up temp files
        try:
            if os.path.exists(temp_webm):
                os.remove(temp_webm)
            if os.path.exists(temp_wav) and temp_wav != temp_webm:
                os.remove(temp_wav)
        except:
            pass
        
        return jsonify({
            'transcription': text,
            'response': response
        })
        
    except Exception as e:
        print(f"❌ Audio processing error: {e}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/text_chat', methods=['POST'])
def text_chat():
    """
    Text-based chat endpoint
    Alternative to voice input
    """
    data = request.json
    user_text = data.get('message', '').strip()
    
    if not user_text:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        print(f"👤 User (via text): {user_text}")
        
        # Get AI response
        if llm:
            # Check if question needs internet
            needs_internet = llm.check_internet_keywords(user_text)
            response = llm.ask(user_text, use_internet=needs_internet)
        else:
            response = "AI is not available right now."
        
        print(f"🤖 {ROBOT_NAME}: {response}")
        
        # Speak response (with servo mouth movement)
        if mouth:
            mouth.speak(response)
        
        return jsonify({'response': response})
        
    except Exception as e:
        print(f"❌ Text chat error: {e}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'robot': ROBOT_NAME,
        'llm_available': llm is not None,
        'tts_available': mouth is not None
    })

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"🌐 {ROBOT_NAME} Web Voice Interface Starting...")
    print(f"{'='*60}")
    print(f"\n📱 Access from your phone:")
    print(f"   http://<raspberry-pi-ip>:{VOICE_PORT}")
    print(f"\n💡 To find your Raspberry Pi IP:")
    print(f"   Run: hostname -I")
    print(f"\n{'='*60}\n")
    
    app.run(host='0.0.0.0', port=VOICE_PORT, debug=False, threaded=True)
