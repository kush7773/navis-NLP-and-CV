"""
Web Voice Interface for Navis Robot
Allows phone to be used as microphone via web browser
"""

from flask import Flask, render_template, request, jsonify
from llm_handler import get_llm
from tts import RobotMouth
from serial_bridge import RobotBridge
import speech_recognition as sr
import io
import os
from pydub import AudioSegment
from config import VOICE_PORT, ROBOT_NAME, SERIAL_PORT, SERIAL_BAUD
import cv2
import numpy as np
import pickle
from datetime import datetime
from face_recognition_utils import (
    save_target_encoding,
    load_target_encoding,
    generate_face_encoding,
    match_target_person,
    detect_and_match_faces
)

app = Flask(__name__)

# Initialize components
try:
    llm = get_llm()
    mouth = RobotMouth()
    bot = RobotBridge(port=SERIAL_PORT, baud_rate=SERIAL_BAUD)
    print(f"✅ Web Voice Interface Initialized")
except Exception as e:
    print(f"⚠️ Warning: Some components failed to initialize: {e}")
    llm = None
    mouth = None
    bot = None

# Camera initialization
try:
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height
    camera_available = True
    print(f"✅ Camera Initialized")
except Exception as e:
    print(f"⚠️ Warning: Camera failed to initialize: {e}")
    camera_available = False
    cap = None

# Person-specific tracking
target_person_encodings = {}  # Dict of encodings: {'front': encoding, 'back': encoding, ...}
target_person_name = "Target Person"
match_threshold = 0.6  # Lower = stricter matching (0.6 is good default)

# Load saved encoding if exists
saved_encodings, saved_name = load_target_encoding()
if saved_encodings and len(saved_encodings) > 0:
    target_person_encodings = saved_encodings
    target_person_name = saved_name
    print(f"✅ Loaded saved person: {target_person_name}")

@app.route('/')
def index():
    """Main voice control page"""
    return render_template('voice_control.html')

@app.route('/hand_control')
def hand_control_page():
    """Hand and arm control page"""
    return render_template('hand_control.html')

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

# ============================================
#   PERSON-SPECIFIC TRACKING ENDPOINTS
# ============================================

@app.route('/upload_person_photo', methods=['POST'])
def upload_person_photo():
    """Upload front/back/side photos to train on person"""
    global target_person_encodings, target_person_name
    
    if 'photo' not in request.files:
        return jsonify({'error': 'No photo uploaded'}), 400
    
    file = request.files['photo']
    name = request.form.get('name', 'Target Person')
    view = request.form.get('view', 'front')  # front, back, side, etc.
    
    try:
        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Generate encoding
        encoding, error = generate_face_encoding(image)
        
        if error:
            return jsonify({'error': error}), 400
        
        # Save encoding with view
        target_person_encodings[view] = encoding
        target_person_name = name
        save_target_encoding(encoding, name, view)
        
        return jsonify({
            'success': True,
            'message': f'Successfully enrolled {name} ({view} view)!',
            'view': view,
            'total_views': len(target_person_encodings)
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/capture_person', methods=['POST'])
def capture_person():
    """Capture current frame and train on person"""
    global target_person_encodings, target_person_name, cap
    
    if not cap or not cap.isOpened():
        return jsonify({'error': 'Camera not available'}), 500
    
    data = request.json if request.json else {}
    name = data.get('name', 'Target Person')
    view = data.get('view', 'front')  # front, back, side, etc.
    
    try:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            return jsonify({'error': 'Failed to capture frame'}), 500
        
        # Generate encoding
        encoding, error = generate_face_encoding(frame)
        
        if error:
            return jsonify({'error': error}), 400
        
        # Save encoding with view
        target_person_encodings[view] = encoding
        target_person_name = name
        save_target_encoding(encoding, name, view)
        
        return jsonify({
            'success': True,
            'message': f'I will remember your {view} view!',
            'view': view,
            'total_views': len(target_person_encodings)
        })
        
    except Exception as e:
        return jsonify({'error': f'Capture failed: {str(e)}'}), 500


@app.route('/clear_person', methods=['POST'])
def clear_person():
    """Clear enrolled person"""
    global target_person_encodings, target_person_name
    
    try:
        target_person_encodings = {}
        target_person_name = "Target Person"
        
        if os.path.exists('target_person.pkl'):
            os.remove('target_person.pkl')
        
        return jsonify({
            'success': True,
            'message': 'Training data cleared'
        })
        
    except Exception as e:
        return jsonify({'error': f'Clear failed: {str(e)}'}), 500


@app.route('/person_status')
def person_status():
    """Get enrollment status"""
    return jsonify({
        'enrolled': len(target_person_encodings) > 0,
        'name': target_person_name if len(target_person_encodings) > 0 else None,
        'views': list(target_person_encodings.keys()),
        'total_views': len(target_person_encodings)
    })

# ============================================
#   HAND & ARM CONTROL ENDPOINTS
# ============================================

@app.route('/hand_control', methods=['POST'])
def hand_control():
    """Control robot hands (open/close)"""
    if not bot:
        return jsonify({'error': 'Robot not connected'}), 500
    
    data = request.json
    command = data.get('command', '').upper()
    
    try:
        if command == 'LC':
            bot.open_left_hand()
            return jsonify({'success': True, 'message': 'Left hand opened'})
        elif command == 'LO':
            bot.close_left_hand()
            return jsonify({'success': True, 'message': 'Left hand closed'})
        elif command == 'RC':
            bot.open_right_hand()
            return jsonify({'success': True, 'message': 'Right hand opened'})
        elif command == 'RO':
            bot.close_right_hand()
            return jsonify({'success': True, 'message': 'Right hand closed'})
        else:
            return jsonify({'error': 'Invalid command'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/wrist_control', methods=['POST'])
def wrist_control():
    """Control robot wrists (0-180 degrees)"""
    if not bot:
        return jsonify({'error': 'Robot not connected'}), 500
    
    data = request.json
    side = data.get('side', '').upper()
    angle = data.get('angle', 90)
    
    try:
        angle = int(angle)
        if angle < 0 or angle > 180:
            return jsonify({'error': 'Angle must be 0-180'}), 400
        
        if side == 'L':
            bot.set_left_wrist(angle)
            return jsonify({'success': True, 'message': f'Left wrist set to {angle}°'})
        elif side == 'R':
            bot.set_right_wrist(angle)
            return jsonify({'success': True, 'message': f'Right wrist set to {angle}°'})
        else:
            return jsonify({'error': 'Invalid side (use L or R)'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid angle value'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/bicep_control', methods=['POST'])
def bicep_control():
    """Control robot biceps (up/down/stop)"""
    if not bot:
        return jsonify({'error': 'Robot not connected'}), 500
    
    data = request.json
    command = data.get('command', '').upper()
    
    try:
        if command == 'BLU':
            bot.left_bicep_up()
            return jsonify({'success': True, 'message': 'Left bicep moving up'})
        elif command == 'BLD':
            bot.left_bicep_down()
            return jsonify({'success': True, 'message': 'Left bicep moving down'})
        elif command == 'BRU':
            bot.right_bicep_up()
            return jsonify({'success': True, 'message': 'Right bicep moving up'})
        elif command == 'BRD':
            bot.right_bicep_down()
            return jsonify({'success': True, 'message': 'Right bicep moving down'})
        elif command == 'BS':
            bot.stop_biceps()
            return jsonify({'success': True, 'message': 'Biceps stopped'})
        else:
            return jsonify({'error': 'Invalid command'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"🌐 {ROBOT_NAME} Web Voice Interface Starting...")
    print(f"{'='*60}")
    print(f"\n📱 Access from your phone:")
    print(f"   http://<raspberry-pi-ip>:{VOICE_PORT}")
    print(f"\n💡 To find your Raspberry Pi IP:")
    print(f"   Run: hostname -I")
    print(f"\n{'='*60}\n")
    
    try:
        app.run(host='0.0.0.0', port=VOICE_PORT, debug=False, threaded=True)
    finally:
        if cap:
            cap.release()
            print("✅ Camera released")
