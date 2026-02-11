"""
Navis Robot - Complete Control Interface
Combines: Camera + Voice + Joystick + Follow Mode
"""

from flask import Flask, render_template, request, jsonify, Response
from llm_handler_updated import get_llm
from tts import RobotMouth
from serial_bridge import RobotBridge
import speech_recognition as sr
import cv2
import io
import os
from pydub import AudioSegment
import numpy as np
import face_recognition
import pickle
from datetime import datetime
from face_recognition_utils import (
    save_target_encoding,
    load_target_encoding,
    generate_face_encoding,
    match_target_person,
    detect_and_match_faces
)
from config import (
    VOICE_PORT, 
    ROBOT_NAME, 
    FOLLOW_COMMANDS, 
    STOP_COMMANDS,
    RASPBERRY_PI_IP,
    SERIAL_PORT,
    SERIAL_BAUD,
    MANUAL_SPEED,
    AUTO_SPEED,
    TURN_SPEED
)
import subprocess
import threading

app = Flask(__name__)

# Initialize components
try:
    llm = get_llm()
    mouth = RobotMouth()
    bot = RobotBridge(port=SERIAL_PORT, baud_rate=SERIAL_BAUD)
    print(f"✅ Complete Control Interface Initialized")
except Exception as e:
    print(f"⚠️ Warning: Some components failed to initialize: {e}")
    llm = None
    mouth = None
    bot = None

# Camera
try:
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    camera_available = True
except:
    camera_available = False
    cap = None

# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global state
follow_mode_active = False
show_face_detection = False

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



def check_follow_command(text):
    """Check if text contains follow or stop command"""
    text_lower = text.lower()
    
    for cmd in FOLLOW_COMMANDS:
        if cmd in text_lower:
            return "FOLLOW"
    
    for cmd in STOP_COMMANDS:
        if cmd in text_lower:
            return "STOP"
    
    return None

def generate_frames():
    """Generate video frames with face detection and follow mode"""
    global show_face_detection, follow_mode_active, target_person_encodings
    
    # Load both face and full-body detectors
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
    upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
    
    while True:
        if not cap:
            break
            
        ret, frame = cap.read()
        if not ret:
            break
        
        # Frame dimensions
        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width // 2
        
        # Tracking variables
        detected_human = False
        target_x = frame_center_x
        target_w = 0
        distance_status = ""
        distance_color = (255, 255, 255)
        
        # Apply human detection if enabled
        if show_face_detection or follow_mode_active:
            
            # ============================================
            #   PERSON-SPECIFIC FACE RECOGNITION
            # ============================================
            if len(target_person_encodings) > 0:
                # Use face recognition to find enrolled person
                face_results = detect_and_match_faces(frame, target_person_encodings, match_threshold)
                
                target_found = False
                
                # Draw boxes for all detected faces
                for result in face_results:
                    x, y, w, h = result['bbox']
                    is_target = result['is_target']
                    confidence = result['confidence']
                    matched_view = result['matched_view']
                    
                    if is_target:
                        # GREEN box for enrolled person
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                        label = f"TARGET ({confidence:.2f})"
                        if matched_view:
                            label += f" - {matched_view.upper()}"
                        cv2.putText(frame, label, (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Update tracking coordinates
                        target_x = x + w // 2
                        target_w = w
                        target_found = True
                        detected_human = True
                    else:
                        # RED box for other people
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, f"OTHER ({confidence:.2f})", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # If target not found by face, try body detection as fallback
                if not target_found:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Try face detection first
                    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
                    upper_bodies = upper_body_cascade.detectMultiScale(gray, 1.1, 3)
                    full_bodies = body_cascade.detectMultiScale(gray, 1.1, 3)
                    
                    # Prioritize detections
                    humans = []
                    if len(faces) > 0:
                        humans = [(x, y, w, h, "FACE") for (x, y, w, h) in faces]
                    elif len(upper_bodies) > 0:
                        humans = [(x, y, w, h, "UPPER BODY") for (x, y, w, h) in upper_bodies]
                    elif len(full_bodies) > 0:
                        humans = [(x, y, w, h, "FULL BODY") for (x, y, w, h) in full_bodies]
                    
                    # Track largest human (assume it's the target who turned away)
                    for (x, y, w, h, label) in humans:
                        if w > target_w:
                            target_x = x + w // 2
                            target_w = w
                            detected_human = True
                            
                            # YELLOW box for body detection (face not visible)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                            cv2.putText(frame, f"{label} (TRACKING)", (x, y-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            else:
                # No person enrolled - use standard body detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                faces = face_cascade.detectMultiScale(gray, 1.2, 5)
                upper_bodies = upper_body_cascade.detectMultiScale(gray, 1.1, 3)
                full_bodies = body_cascade.detectMultiScale(gray, 1.1, 3)
                
                # Prioritize detections
                humans = []
                if len(faces) > 0:
                    humans = [(x, y, w, h, "FACE") for (x, y, w, h) in faces]
                elif len(upper_bodies) > 0:
                    humans = [(x, y, w, h, "UPPER BODY") for (x, y, w, h) in upper_bodies]
                elif len(full_bodies) > 0:
                    humans = [(x, y, w, h, "FULL BODY") for (x, y, w, h) in full_bodies]
                
                # Draw bounding boxes and track largest human
                for (x, y, w, h, label) in humans:
                    detected_human = True
                    if w > target_w:
                        target_x = x + w // 2
                        target_w = w
                    
                    # BLUE box for standard detection
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # ============================================
        #   FOLLOW MODE MOTOR CONTROL
        # ============================================
        if follow_mode_active and detected_human and bot:
            error = target_x - frame_center_x
            threshold = 50  # Horizontal deadzone
            
            # Distance thresholds (based on bounding box width)
            OPTIMAL_MIN_SIZE = 80   # Too close if bigger
            OPTIMAL_MAX_SIZE = 150  # Too far if smaller
            STOP_DISTANCE = 200     # Emergency stop if very close
            
            if target_w > STOP_DISTANCE:
                # TOO CLOSE - STOP
                bot.stop()
                distance_status = "TOO CLOSE - STOPPED"
                distance_color = (0, 0, 255)  # Red
                
            elif target_w > OPTIMAL_MIN_SIZE:
                # GOOD DISTANCE - Only turn to center
                if abs(error) < threshold:
                    bot.stop()
                    distance_status = "OPTIMAL - CENTERED"
                    distance_color = (0, 255, 0)  # Green
                elif error > 0:
                    bot.drive(-TURN_SPEED, TURN_SPEED)
                    distance_status = "TURNING RIGHT"
                    distance_color = (255, 255, 0)  # Yellow
                else:
                    bot.drive(TURN_SPEED, -TURN_SPEED)
                    distance_status = "TURNING LEFT"
                    distance_color = (255, 255, 0)  # Yellow
                    
            elif target_w < OPTIMAL_MAX_SIZE:
                # TOO FAR - Move forward
                if abs(error) < threshold:
                    bot.drive(AUTO_SPEED, AUTO_SPEED)
                    distance_status = "MOVING FORWARD"
                    distance_color = (0, 255, 255)  # Cyan
                elif error > 0:
                    bot.drive(AUTO_SPEED, AUTO_SPEED // 2)
                    distance_status = "FORWARD + RIGHT"
                    distance_color = (0, 255, 255)  # Cyan
                else:
                    bot.drive(AUTO_SPEED // 2, AUTO_SPEED)
                    distance_status = "FORWARD + LEFT"
                    distance_color = (0, 255, 255)  # Cyan
                    
            else:
                # IN OPTIMAL RANGE - Fine adjustments
                if abs(error) < threshold:
                    bot.stop()
                    distance_status = "OPTIMAL - CENTERED"
                    distance_color = (0, 255, 0)  # Green
                elif error > 0:
                    bot.drive(-TURN_SPEED, TURN_SPEED)
                    distance_status = "ADJUSTING RIGHT"
                    distance_color = (255, 255, 0)  # Yellow
                else:
                    bot.drive(TURN_SPEED, -TURN_SPEED)
                    distance_status = "ADJUSTING LEFT"
                    distance_color = (255, 255, 0)  # Yellow
        
        elif follow_mode_active and not detected_human and bot:
            # Lost target - stop and search
            bot.stop()
            distance_status = "SEARCHING..."
            distance_color = (255, 165, 0)  # Orange
        
        # ============================================
        #   VISUAL OVERLAY
        # ============================================
        if follow_mode_active:
            # Show follow mode status
            cv2.putText(frame, "FOLLOW MODE: ON", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if detected_human and distance_status:
                # Show distance status
                cv2.putText(frame, distance_status, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, distance_color, 2)
                # Show distance indicator
                distance_text = f"Distance: {target_w}px"
                cv2.putText(frame, distance_text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                tracking_status = "SEARCHING..."
                cv2.putText(frame, tracking_status, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        
        # Person enrollment status
        if len(target_person_encodings) > 0:
            enrolled_text = f"Enrolled: {target_person_name} ({len(target_person_encodings)} views)"
            cv2.putText(frame, enrolled_text, (10, frame_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 136), 1)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Main control interface"""
    return render_template('complete_control.html', camera_available=camera_available)

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
    
    # Map commands to motor speeds
    if command == 'F':  # Forward
        bot.drive(MANUAL_SPEED, MANUAL_SPEED)
    elif command == 'B':  # Back
        bot.drive(-MANUAL_SPEED, -MANUAL_SPEED)
    elif command == 'L':  # Left
        bot.drive(-MANUAL_SPEED, MANUAL_SPEED)
    elif command == 'R':  # Right
        bot.drive(MANUAL_SPEED, -MANUAL_SPEED)
    elif command == 'S':  # Stop
        bot.stop()
    else:
        return jsonify({'error': 'Invalid command'}), 400
    
    return jsonify({'status': 'ok', 'command': command})

@app.route('/audio_upload', methods=['POST'])
def audio_upload():
    """Voice control endpoint"""
    audio_file = request.files.get('audio')
    
    if not audio_file:
        return jsonify({'error': 'No audio received'}), 400
    
    try:
        temp_webm = '/tmp/user_audio.webm'
        temp_wav = '/tmp/user_audio.wav'
        audio_file.save(temp_webm)
        
        try:
            audio = AudioSegment.from_file(temp_webm, format="webm")
            audio.export(temp_wav, format="wav")
        except:
            temp_wav = temp_webm
        
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(temp_wav) as source:
            audio_data = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio_data)
            print(f"👤 User: {text}")
        except sr.UnknownValueError:
            return jsonify({'error': 'Could not understand audio'}), 400
        except sr.RequestError:
            return jsonify({'error': 'Speech recognition unavailable'}), 500
        
        # Check for follow/stop commands
        command = check_follow_command(text)
        
        if command == "FOLLOW":
            global follow_mode_active, show_face_detection
            follow_mode_active = True
            show_face_detection = True
            response = "Follow mode activated! I will track and follow you."
            
        elif command == "STOP":
            follow_mode_active = False
            if bot:
                bot.stop()
            response = "Stopping. Follow mode deactivated."
            
        else:
            # Normal AI conversation - optimized for speed
            if llm:
                try:
                    # Use shorter responses for faster processing
                    response = llm.ask(text + " (answer in 20 words or less)")
                except:
                    response = "Sorry, I'm having trouble thinking right now."
            else:
                response = "AI is not available."
        
        print(f"🤖 {ROBOT_NAME}: {response}")
        
        if mouth:
            mouth.speak(response)
        
        # Cleanup
        try:
            if os.path.exists(temp_webm):
                os.remove(temp_webm)
            if os.path.exists(temp_wav) and temp_wav != temp_webm:
                os.remove(temp_wav)
        except:
            pass
        
        return jsonify({
            'transcription': text,
            'response': response,
            'follow_active': follow_mode_active
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/text_chat', methods=['POST'])
def text_chat():
    """Text-based chat"""
    data = request.json
    user_text = data.get('message', '').strip()
    
    if not user_text:
        return jsonify({'error': 'No message'}), 400
    
    try:
        print(f"👤 User: {user_text}")
        
        command = check_follow_command(user_text)
        
        if command == "FOLLOW":
            global follow_mode_active, show_face_detection
            follow_mode_active = True
            show_face_detection = True
            response = "Follow mode activated!"
            
        elif command == "STOP":
            follow_mode_active = False
            if bot:
                bot.stop()
            response = "Stopped."
            
        else:
            if llm:
                response = llm.ask(user_text)
            else:
                response = "AI unavailable."
        
        print(f"🤖 {ROBOT_NAME}: {response}")
        
        if mouth:
            mouth.speak(response)
        
        return jsonify({
            'response': response,
            'follow_active': follow_mode_active
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    """Get system status"""
    return jsonify({
        'robot': ROBOT_NAME,
        'follow_active': follow_mode_active,
        'face_detection': show_face_detection,
        'camera_available': camera_available,
        'llm_available': llm is not None,
        'motors_available': bot is not None
    })

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"🤖 {ROBOT_NAME} Complete Control Interface")
    print(f"{'='*60}")
    print(f"\n📱 Access: http://{RASPBERRY_PI_IP}:{VOICE_PORT}")
    print(f"\n✨ Features:")
    print(f"   📹 Live Camera Feed")
    print(f"   🎤 Voice Control")
    print(f"   🕹️  Joystick Controls")
    print(f"   👤 Face Detection")
    print(f"   🎯 Follow Mode")
    print(f"\n{'='*60}\n")
    
    try:
        app.run(host='0.0.0.0', port=VOICE_PORT, debug=False, threaded=True)
    finally:
        if bot:
            bot.close()
        if cap:
            cap.release()

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
