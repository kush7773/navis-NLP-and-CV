"""
Voice-Activated Follow Mode Integration
Combines voice commands with existing CV face tracking
"""

from flask import Flask, render_template, request, jsonify, Response
from llm_handler_updated import get_llm
from tts import RobotMouth
from serial_bridge import RobotBridge
import speech_recognition as sr
import io
import os
from pydub import AudioSegment
from config import (
    VOICE_PORT, 
    ROBOT_NAME, 
    FOLLOW_COMMANDS, 
    STOP_COMMANDS,
    RASPBERRY_PI_IP,
    SERIAL_PORT,
    SERIAL_BAUD,
    MANUAL_SPEED,
    TURN_SPEED
)
import subprocess
import signal
import threading
import time
import cv2
import numpy as np
import pickle
from datetime import datetime
from face_recognition_utils import (
    generate_face_encoding,
    save_target_encoding,
    load_target_encoding
)

app = Flask(__name__)

# Initialize components
try:
    llm = get_llm()
    mouth = RobotMouth()
    bot = RobotBridge(port=SERIAL_PORT, baud_rate=SERIAL_BAUD)
    print(f"✅ Voice + Follow Integration Initialized")
except Exception as e:
    print(f"⚠️ Warning: Some components failed to initialize: {e}")
    llm = None
    mouth = None
    bot = None

# Camera initialization
try:
    # Try indices 0, 1, 2 to find working camera
    video_source = 0
    cap = None
    for i in range(3):
        print(f"📷 Testing camera index {i}...")
        temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            ret, _ = temp_cap.read()
            if ret:
                cap = temp_cap
                video_source = i
                print(f"✅ Camera found at index {i}")
                break
            else:
                temp_cap.release()
        
    if cap is None:
        raise Exception("No working camera found on indices 0-2")
        
    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height
    camera_available = True
    print(f"✅ Camera Initialized on index {video_source}")
except Exception as e:
    print(f"⚠️ Warning: Camera failed to initialize: {e}")
    camera_available = False
    cap = None
    print(f"⚠️ Warning: Camera failed to initialize: {e}")
    camera_available = False
    cap = None

# Global state
show_face_detection = False
frames_lock = threading.Lock()
follow_mode_active = False
target_person_encodings = {}
target_person_name = ""

def generate_frames():
    """Video streaming generator function"""
    global cap, show_face_detection, target_person_encodings
    
    while True:
        if not camera_available or cap is None:
            break
            
        with frames_lock:
            success, frame = cap.read()
            
        if not success:
            break
            
        # Add overlay and tracking if enabled
        if show_face_detection or follow_mode_active:
             # Skip frames for detection performance (process every 3rd frame)
            if not hasattr(generate_frames, "frame_count"):
                generate_frames.frame_count = 0
            
            generate_frames.frame_count += 1
            
            # Reset motor values
            left_mtr, right_mtr = 0, 0
            
            if generate_frames.frame_count % 3 == 0:
                try:
                    # Resize for faster detection
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                    
                    # Detect faces using Cascade (Faster than HOG/CNN on Pi)
                    faces = face_cascade.detectMultiScale(gray_small, 1.1, 4)
                    
                    # Store for drawing
                    scaled_faces = []
                    
                    for (x, y, w, h) in faces:
                        # Scale up to original size
                        x *= 4
                        y *= 4
                        w *= 4
                        h *= 4
                        scaled_faces.append((x, y, w, h))
                        
                        # Tracking Logic
                        if follow_mode_active and bot:
                            cx = x + w // 2
                            # Center is 320 for 640x480
                            error = cx - 320
                            
                            color = (0, 255, 0) # Green = Tracking
                            
                            if abs(error) > DEAD_ZONE:
                                if error > 0: # Right
                                    left_mtr, right_mtr = TURN_SPEED_TRACK, -TURN_SPEED_TRACK
                                else: # Left
                                    left_mtr, right_mtr = -TURN_SPEED_TRACK, TURN_SPEED_TRACK
                            else:
                                if w < 100: # Too far
                                    left_mtr, right_mtr = AUTO_SPEED, AUTO_SPEED
                                elif w > 200: # Too close
                                    left_mtr, right_mtr = -AUTO_SPEED, -AUTO_SPEED
                                else:
                                    # Perfect distance
                                    left_mtr, right_mtr = 0, 0
                            
                            # Execute drive (only for biggest face/first face)
                            bot.drive(left_mtr, right_mtr)
                            break # Only track one face
                        
                    generate_frames.last_faces = scaled_faces

                except Exception as e:
                    print(f"Detection error: {e}")
            
            # Draw persistent boxes
            if hasattr(generate_frames, "last_faces"):
                for (x, y, w, h) in generate_frames.last_faces:
                    color = (0, 255, 0) if follow_mode_active else (255, 0, 0)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    if follow_mode_active:
                         cv2.putText(frame, "TRACKING", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Stop if no faces found in tracking mode (safety)
        if follow_mode_active and bot and generate_frames.frame_count % 3 == 0 and (not hasattr(generate_frames, "last_faces") or len(generate_frames.last_faces) == 0):
            bot.stop()

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Tracking Configuration
AUTO_SPEED = 85
TURN_SPEED_TRACK = 100
DEAD_ZONE = 60
TRAINER_FILE = 'trainer.yml'

# Initialize Trackers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
model_trained = False

# Load model if exists
if os.path.exists(TRAINER_FILE):
    try:
        recognizer.read(TRAINER_FILE)
        model_trained = True
        print(f"✅ Loaded face model from {TRAINER_FILE}")
    except:
        print("⚠️ Could not load trainer.yml")

def start_follow_mode():
    """Activate follow mode"""
    global follow_mode_active
    follow_mode_active = True
    print("✅ Follow mode activated")
    return True

def stop_follow_mode():
    """Deactivate follow mode"""
    global follow_mode_active
    follow_mode_active = False
    if bot:
        bot.stop()
    print("⏸️ Follow mode deactivated")
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



@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    if not camera_available:
        return "Camera not available", 404
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_face_detection', methods=['POST'])
def toggle_face_detection():
    """Toggle face detection overlay"""
    global show_face_detection
    show_face_detection = not show_face_detection
    return jsonify({'enabled': show_face_detection})

@app.route('/manual_control', methods=['POST'])
def manual_control():
    """Handle joystick/manual controls: F, B, L, R, S"""
    data = request.json
    command = data.get('command', '').upper()
    
    if not bot:
        return jsonify({'error': 'Robot not connected'}), 500
    
    try:
        # Map commands to motor speeds
        if command == 'F':  # Forward
            bot.drive(MANUAL_SPEED, MANUAL_SPEED)
        elif command == 'B':  # Back
            bot.drive(-MANUAL_SPEED, -MANUAL_SPEED)
        elif command == 'L':  # Left
            bot.drive(-TURN_SPEED, TURN_SPEED)
        elif command == 'R':  # Right
            bot.drive(TURN_SPEED, -TURN_SPEED)
        elif command == 'S':  # Stop
            bot.stop()
        else:
            return jsonify({'error': 'Invalid command'}), 400
        
        return jsonify({'status': 'ok', 'command': command})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

# ============================================
#   PERSON TRACKING ENDPOINTS
# ============================================

@app.route('/upload_person_photo', methods=['POST'])
def upload_person_photo():
    """Upload front/back/side photos to train on person"""
    global target_person_encodings, target_person_name
    
    data = request.json
    image_data = data.get('image')  # base64 encoded
    view_name = data.get('view', 'front')  # front, back, left, right
    person_name = data.get('name', 'Target Person')
    
    if not image_data:
        return jsonify({'error': 'No image data'}), 400
    
    try:
        # Decode base64 image
        import base64
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Generate face encoding
        encoding = generate_face_encoding(img)
        
        if encoding is None:
            return jsonify({'error': 'No face detected in image'}), 400
        
        # Store encoding
        target_person_encodings[view_name] = encoding
        target_person_name = person_name
        
        # Save to disk (save individual encoding with view)
        save_target_encoding(encoding, person_name, view_name)
        
        return jsonify({
            'success': True,
            'message': f'{view_name.capitalize()} view saved',
            'total_views': len(target_person_encodings)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/capture_person', methods=['POST'])
def capture_person():
    """Capture current frame and train on person"""
    global target_person_encodings, target_person_name, cap
    
    if not camera_available or cap is None:
        return jsonify({'error': 'Camera not available'}), 500
    
    data = request.json
    view_name = data.get('view', 'front')
    person_name = data.get('name', 'Target Person')
    
    try:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            return jsonify({'error': 'Failed to capture frame'}), 500
        
        # Generate encoding
        encoding = generate_face_encoding(frame)
        
        if encoding is None:
            return jsonify({'error': 'No face detected in frame'}), 400
        
        # Store encoding
        target_person_encodings[view_name] = encoding
        target_person_name = person_name
        
        # Save to disk (save individual encoding with view)
        save_target_encoding(encoding, person_name, view_name)
        
        return jsonify({
            'success': True,
            'message': f'{view_name.capitalize()} view captured',
            'total_views': len(target_person_encodings)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/clear_person', methods=['POST'])
def clear_person():
    """Clear all person training data"""
    global target_person_encodings, target_person_name
    
    target_person_encodings = {}
    target_person_name = ""
    
    # Delete saved file
    try:
        if os.path.exists('target_person.pkl'):
            os.remove('target_person.pkl')
    except:
        pass
    
    return jsonify({
        'success': True,
        'message': 'Training data cleared'
    })


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


@app.route('/hand_control_page')
def hand_control_page():
    """Hand and arm control page"""
    return render_template('hand_control.html')

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
        # Run with SSL (HTTPS) for microphone access
        # Requires cert.pem and key.pem (run generate_cert.sh)
        if os.path.exists('cert.pem') and os.path.exists('key.pem'):
            print(f"🔐 Starting with SSL (HTTPS) using cert.pem/key.pem")
            app.run(host='0.0.0.0', port=VOICE_PORT, debug=False, threaded=True, ssl_context=('cert.pem', 'key.pem'))
        else:
            print(f"⚠️ SSL Certificates not found (cert.pem, key.pem)")
            print(f"   Microphone might not work on remote devices.")
            print(f"   Running in HTTP mode. Run ./generate_cert.sh to enable HTTPS.")
            app.run(host='0.0.0.0', port=VOICE_PORT, debug=False, threaded=True)
    finally:
        # Cleanup on exit
        if bot:
            bot.stop()
        if cap:
            cap.release()
