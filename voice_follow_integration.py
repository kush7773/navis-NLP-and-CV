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
import face_recognition
from face_recognition_utils import (
    generate_face_encoding,
    save_target_encoding,
    load_target_encoding,
    match_target_person
)
import mediapipe as mp

# Configuration
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# Distance Calibration (Approximate)
# 0.2 depth_percent ~= 20cm (Shoulders fill ~80% of screen)
DEPTH_THRESHOLD_NEAR = 0.15    # Move back if closer than this
DEPTH_THRESHOLD_FAR = 0.25     # Move forward if farther than this
CENTER_TOLERANCE = 0.15        # 15% tolerance for center detection

# --- MediaPipe Pose Tracker ---
class PoseTracker:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,              # Lighter model for speed
            smooth_landmarks=True,
            min_detection_confidence=0.4,    # Lower threshold for speed
            min_tracking_confidence=0.4
        )

    def analyze_pose(self, frame):
        """
        Analyze person position and depth from pose landmarks
        Returns: (position, depth, depth_percent, results)
        """
        h, w, c = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None, None, None, results
        
        # Get key landmarks for position analysis
        landmarks = results.pose_landmarks.landmark
        
        # Use nose (landmark 0) and shoulders for position/depth detection
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        # Check confidence
        if nose.visibility < 0.5:
            return None, None, None, results
        
        # --- Calculate Horizontal Position (Left/Center/Right) ---
        nose_x_norm = nose.x
        
        if nose_x_norm < (0.5 - CENTER_TOLERANCE):
            position = 'left'
        elif nose_x_norm > (0.5 + CENTER_TOLERANCE):
            position = 'right'
        else:
            position = 'center'
        
        # --- Calculate Depth (Near/Medium/Far) ---
        if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
            shoulder_width = abs(right_shoulder.x - left_shoulder.x)
            shoulder_width_norm = shoulder_width
        else:
            shoulder_width_norm = 0.3
        
        # Estimate Distance (cm)
        # K = 16 (derived from 20cm target at 0.8 width)
        # Avoid division by zero
        if shoulder_width_norm > 0:
            est_dist_cm = 16.0 / shoulder_width_norm
        else:
            est_dist_cm = 100.0
            
        # Reverse: larger width = closer (keep for legacy compatibility if needed, but we use cm now)
        depth_percent = 1.0 - np.clip(shoulder_width_norm, 0, 1)
        
        # Determine zone based on cm
        if est_dist_cm < 18:
            depth = 'near' # Too close (<18cm)
        elif est_dist_cm > 28:
            depth = 'far'  # Too far (>28cm)
        else:
            depth = 'medium' # Good (18-28cm)
            
        return position, depth, est_dist_cm, results

# Initialize Tracker
pose_tracker = PoseTracker()

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
manual_mode_active = False

def generate_frames():
    """Video streaming generator function with Hybrid Tracking"""
    global cap, show_face_detection, target_person_encodings, target_person_name, manual_mode_active
    
    # Tracking State
    frame_count = 0
    last_driving_command = "STOP"
    search_mode = False
    is_target_provisional = False
    last_face_seen_time = 0
    
    # Ensure camera is initialized
    if not 'cap' in globals() or cap is None:
        print("📷 Camera not initialized, attempting to restart...")
        # Try to re-init (simplified)
        try:
            cap = cv2.VideoCapture(0)
            cap.set(3, 640)
            cap.set(4, 480)
        except:
            pass

    while True:
        if cap is None or not cap.isOpened():
            # Try to reconnect occasionally
            time.sleep(1)
            continue
            
        with frames_lock:
            success, frame = cap.read()
            
        if not success:
            continue

        frame_count += 1
        
        # Add overlay and tracking if enabled
        if show_face_detection or follow_mode_active:
            
            # 1. MediaPipe Pose Analysis (Always run if tracking to get depth/pos)
            # Resize for speed? PoseTracker can handle it.
            # Convert to RGB for MediaPipe
            mp_pos, mp_depth, mp_dist_cm, mp_results = pose_tracker.analyze_pose(frame)
            
            # Draw Pose Landmarks
            if mp_results and mp_results.pose_landmarks:
                pose_tracker.mp_drawing.draw_landmarks(
                    frame,
                    mp_results.pose_landmarks,
                    pose_tracker.mp_holistic.POSE_CONNECTIONS,
                    pose_tracker.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    pose_tracker.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
                )
                if mp_dist_cm:
                    cv2.putText(frame, f"Dist: {int(mp_dist_cm)}cm", (10, 470), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 2. Face Detection (Identity Check)
            # Only run every 3rd frame to save CPU
            if frame_count % 3 == 0:
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                try:
                    faces = face_cascade.detectMultiScale(
                        cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY), 
                        1.1, 4, minSize=(30, 30)
                    )
                    
                    found_target_face = False
                    
                    for (x, y, w, h) in faces:
                        # Draw generic box
                        # Scale up
                        top, right, bottom, left = y*2, (x+w)*2, (y+h)*2, x*2
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        
                        # Check Identity if we have a target
                        if target_person_encodings:
                            # Extract face
                            face_img = rgb_small[y:y+h, x:x+w]
                            if face_img.size > 0:
                                face_img = np.ascontiguousarray(face_img)
                                encodings = face_recognition.face_encodings(face_img)
                                if encodings:
                                    match, conf, _ = match_target_person(encodings[0], target_person_encodings, 0.65)
                                    if match:
                                        found_target_face = True
                                        cv2.putText(frame, f"TARGET {conf:.2f}", (left, top-10), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    if found_target_face:
                        is_target_provisional = True
                        last_face_seen_time = time.time()  # Refresh lock
                    else:
                        is_target_provisional = False
                        
                except Exception as e:
                    print(f"Face Error: {e}")

            # 3. Driving Logic (Hybrid with Persistence)
            if follow_mode_active and bot and not manual_mode_active:
                cmd = "STOP"
                
                # Default speeds
                fwd_speed = AUTO_SPEED
                turn_speed = TURN_SPEED
                
                # Persistence Check
                time_since_face = time.time() - last_face_seen_time
                is_locked = (time_since_face < 3.5)  # 3.5 second buffer
                
                if is_locked and mp_pos:
                    # Calculate Dynamic Speed (Proportional Control)
                    # Target Dist ~22cm. Error = abs(dist - 22)
                    if mp_dist_cm:
                        dist_error = abs(mp_dist_cm - 22)
                        # Forward Proportional: 80 to 85 (User Specified)
                        p_speed = 80 + (dist_error * 1.0)
                        fwd_speed = int(np.clip(p_speed, 80, 85))
                        
                        # Backward Proportional: 55 to 65 (User Specified)
                        # Closer we are, faster we back up? Or standard P?
                        # dist_error is high when very close (<18). 
                        # Let's map error 0-10 -> speed 55-65
                        p_back = 55 + (dist_error * 1.0)
                        back_speed = int(np.clip(p_back, 55, 65))
                    else:
                        fwd_speed = 85
                        back_speed = 60
                    
                    # Visual feedback for sticky tracking
                    # Visual feedback for sticky tracking
                    if not is_target_provisional:
                        cv2.putText(frame, f"LOCKED ({3.5-time_since_face:.1f}s)", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # Use MediaPipe for control
                    if mp_depth == 'far':
                         if mp_pos == 'center':
                             bot.drive(fwd_speed, fwd_speed)
                             cmd = f"FORWARD ({fwd_speed})"
                         elif mp_pos == 'left':
                             bot.drive(int(fwd_speed*0.5), fwd_speed)
                             cmd = "FWD-LEFT"
                         elif mp_pos == 'right':
                             bot.drive(fwd_speed, int(fwd_speed*0.5))
                             cmd = "FWD-RIGHT"
                    elif mp_depth == 'near':
                         bot.drive(-back_speed, -back_speed)
                         cmd = f"BACKWARD ({back_speed})"
                    else: # Medium
                         if mp_pos == 'center':
                             bot.stop()
                             cmd = "STOP (OPTIMAL)"
                         elif mp_pos == 'left':
                             bot.drive(-turn_speed, turn_speed) # Rotate
                             cmd = "ROTATE-LEFT" 
                         elif mp_pos == 'right':
                             bot.drive(turn_speed, -turn_speed) # Rotate
                             cmd = "ROTATE-RIGHT"
                else:
                    # Not locked or no body found -> Stop
                    # Only stop if we were moving
                    if last_driving_command != "STOP":
                        bot.stop()
                    cmd = "STOP (NO TARGET)"

                # --- DEBUG LOGGING ---
                if frame_count % 10 == 0:
                     status_msg = f"Follow:{follow_mode_active} | Locked:{is_locked} | Dist:{int(mp_dist_cm) if mp_dist_cm else 0}cm | CMD:{cmd}"
                     print(f"🔍 DEBUG: {status_msg}")
                     
                # --- VISUAL STATUS INDICATORS ---
                # Follow Mode Status
                status_color = (0, 255, 0) if follow_mode_active else (0, 0, 255)
                status_text = "FOLLOW: ON" if follow_mode_active else "FOLLOW: OFF"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                # Command Status
                cv2.putText(frame, f"CMD: {cmd}", (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                if cmd != last_driving_command:
                    # print(f"🚗 Drive: {cmd} (Pos: {mp_pos}, Depth: {mp_depth})")
                    last_driving_command = cmd

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

# Load target person data
try:
    target_person_encodings, target_person_name = load_target_encoding()
    if target_person_encodings:
        print(f"✅ Loaded target person: {target_person_name} ({len(target_person_encodings)} views)")
    else:
        print("ℹ️ No target person trained yet.")
        target_person_encodings = {}
        target_person_name = ""
except Exception as e:
    print(f"⚠️ Error loading training data: {e}")
    target_person_encodings = {}
    target_person_name = ""

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
        'hybrid_running': follow_mode_active # Simplified since it's same process
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
    
    # MANUAL OVERRIDE: Any manual command disables follow mode
    global follow_mode_active
    if follow_mode_active:
        follow_mode_active = False
        print("⚠️ Manual Override: Follow Mode Deactivated")
    
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
        encoding, error = generate_face_encoding(img)
        
        if error:
            return jsonify({'error': error}), 400
        
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
        
        # Generate face encoding
        encoding, error = generate_face_encoding(frame)
        
        if error:
            return jsonify({'error': error}), 400
        
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
