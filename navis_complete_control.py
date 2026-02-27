"""
Navis Robot - Complete Control Interface
Unified: Camera + Voice + Joystick + Follow Mode + Hand/Arm Control + Model Training
"""

from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from llm_handler_updated import get_llm
import os
import io
import subprocess
import threading
from datetime import datetime

# Graceful imports for hardware-dependent modules
try:
    from tts import RobotMouth
except ImportError:
    print("⚠️ TTS not available (pyttsx3 missing)")
    RobotMouth = None

try:
    from serial_bridge import RobotBridge
except ImportError:
    print("⚠️ Serial bridge not available (pyserial missing)")
    RobotBridge = None

try:
    import speech_recognition as sr
except ImportError:
    print("⚠️ Speech recognition not available")
    sr = None

try:
    import cv2
except ImportError:
    print("⚠️ OpenCV not available")
    cv2 = None

try:
    from pydub import AudioSegment
except ImportError:
    print("⚠️ pydub not available")
    AudioSegment = None

try:
    import numpy as np
except ImportError:
    print("⚠️ numpy not available")
    np = None

try:
    import face_recognition
except ImportError:
    print("⚠️ face_recognition not available")
    face_recognition = None

try:
    import pickle
except ImportError:
    pickle = None

try:
    from face_recognition_utils import (
        save_target_encoding,
        load_target_encoding,
        generate_face_encoding,
        match_target_person,
        detect_and_match_faces
    )
except ImportError:
    print("⚠️ face_recognition_utils not available")
    save_target_encoding = None
    def load_target_encoding(): return {}, ""
    generate_face_encoding = None
    match_target_person = None
    detect_and_match_faces = None

from config import (
    VOICE_PORT, 
    ROBOT_NAME, 
    FOLLOW_COMMANDS, 
    STOP_COMMANDS,
    RASPBERRY_PI_IP,
    SERIAL_PORT,
    MEGA_PORT,
    SERIAL_BAUD,
    MANUAL_SPEED,
    AUTO_SPEED,
    TURN_SPEED
)

# --- MediaPipe Pose Tracker (from working reference) ---
try:
    import mediapipe as mp

    class PoseTracker:
        def __init__(self):
            self.mp_holistic = mp.solutions.holistic
            self.mp_drawing = mp.solutions.drawing_utils
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=0,
                smooth_landmarks=True,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4
            )

        def analyze_pose(self, frame):
            """Returns: (position, depth, distance_cm, results)"""
            h, w, c = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(frame_rgb)

            if not results.pose_landmarks:
                return None, None, None, results

            landmarks = results.pose_landmarks.landmark
            nose = landmarks[0]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]

            if nose.visibility < 0.5:
                return None, None, None, results

            # Horizontal position
            CENTER_TOLERANCE = 0.15
            if nose.x < (0.5 - CENTER_TOLERANCE):
                position = 'left'
            elif nose.x > (0.5 + CENTER_TOLERANCE):
                position = 'right'
            else:
                position = 'center'

            # Depth from shoulder width
            if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                shoulder_width = abs(right_shoulder.x - left_shoulder.x)
            else:
                shoulder_width = 0.3

            est_dist_cm = 16.0 / shoulder_width if shoulder_width > 0 else 100.0

            if est_dist_cm < 18:
                depth = 'near'
            elif est_dist_cm > 28:
                depth = 'far'
            else:
                depth = 'medium'

            return position, depth, est_dist_cm, results

    pose_tracker = PoseTracker()
    print("✅ MediaPipe Pose Tracker initialized")
except (ImportError, AttributeError, Exception) as e:
    print(f"⚠️ MediaPipe not available ({e}) — follow mode will use Haar cascade fallback")
    pose_tracker = None

app = Flask(__name__, static_folder='static')

# Initialize components
llm = None
mouth = None
bot = None

try:
    llm = get_llm()
    print("✅ LLM initialized")
except Exception as e:
    print(f"⚠️ LLM init failed: {e}")

try:
    if RobotMouth:
        mouth = RobotMouth()
        print("✅ TTS initialized")
except Exception as e:
    print(f"⚠️ TTS init failed: {e}")

try:
    if RobotBridge:
        bot = RobotBridge(esp32_port=SERIAL_PORT, mega_port=MEGA_PORT, baud_rate=SERIAL_BAUD)
        print("✅ Motor bridge initialized")
except Exception as e:
    print(f"⚠️ Motor bridge init failed: {e}")

# Camera
try:
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    camera_available = True
except:
    camera_available = False
    cap = None

# Face detection (Haar cascade for identity check only)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') if cv2 else None

# Global state
follow_mode_active = False
show_face_detection = False
active_drive_cmd = "STOP"
active_arm_cmd = "IDLE"

# Person-specific tracking
target_person_encodings = {}
target_person_name = "Target Person"
match_threshold = 0.6

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
    """Video streaming with MediaPipe Pose tracking (from working reference)"""
    global show_face_detection, follow_mode_active, target_person_encodings
    global active_drive_cmd, active_arm_cmd
    
    import time as _time
    
    frame_count = 0
    last_driving_command = "STOP"
    last_face_seen_time = 0
    is_target_provisional = False
    
    while True:
        if not cap:
            break
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]
        frame_count += 1
        
        if show_face_detection or follow_mode_active:
            # 1. MediaPipe Pose (every frame, fast)
            mp_pos, mp_depth, mp_dist_cm, mp_results = None, None, None, None
            if pose_tracker:
                mp_pos, mp_depth, mp_dist_cm, mp_results = pose_tracker.analyze_pose(frame)
                if mp_results and mp_results.pose_landmarks:
                    pose_tracker.mp_drawing.draw_landmarks(
                        frame, mp_results.pose_landmarks,
                        pose_tracker.mp_holistic.POSE_CONNECTIONS,
                        pose_tracker.mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                        pose_tracker.mp_drawing.DrawingSpec(color=(255,0,0), thickness=1))
                    if mp_dist_cm:
                        cv2.putText(frame, f"Dist: {int(mp_dist_cm)}cm", (10, frame_height-50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            
            # 2. Face ID check (every 3rd frame, 0.5x)
            if frame_count % 3 == 0 and face_cascade is not None:
                small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
                gray_s = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                try:
                    faces = face_cascade.detectMultiScale(gray_s, 1.1, 4, minSize=(30,30))
                    found_target = False
                    for (x,y,w,h) in faces:
                        t,r,b,l = y*2,(x+w)*2,(y+h)*2,x*2
                        cv2.rectangle(frame,(l,t),(r,b),(0,0,255),2)
                        if target_person_encodings and face_recognition:
                            rgb_s = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                            face_img = rgb_s[y:y+h, x:x+w]
                            if face_img.size > 0:
                                face_img = np.ascontiguousarray(face_img)
                                encs = face_recognition.face_encodings(face_img)
                                if encs and match_target_person:
                                    match, conf, _ = match_target_person(encs[0], target_person_encodings, match_threshold)
                                    if match:
                                        found_target = True
                                        cv2.putText(frame, f"TARGET {conf:.2f}", (l,t-10),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                                        cv2.rectangle(frame,(l,t),(r,b),(0,255,0),2)
                    if found_target:
                        is_target_provisional = True
                        last_face_seen_time = _time.time()
                    else:
                        is_target_provisional = False
                except:
                    pass
            
            # 3. Driving Logic (3-zone from working reference)
            if follow_mode_active and bot:
                cmd = "STOP"
                fwd_speed = AUTO_SPEED
                back_speed = int(AUTO_SPEED * 0.7)
                turn_speed = TURN_SPEED
                
                time_since_face = _time.time() - last_face_seen_time
                is_locked = (time_since_face < 3.5)
                
                if is_locked and mp_pos:
                    if mp_dist_cm:
                        dist_error = abs(mp_dist_cm - 22)
                        fwd_speed = int(np.clip(80 + dist_error, 80, 85))
                        back_speed = int(np.clip(55 + dist_error, 55, 65))
                    else:
                        fwd_speed, back_speed = 85, 60
                    
                    if not is_target_provisional:
                        cv2.putText(frame, f"LOCKED ({3.5-time_since_face:.1f}s)", (50,50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    
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
                    else:
                        if mp_pos == 'center':
                            bot.stop()
                            cmd = "STOP (OPTIMAL)"
                        elif mp_pos == 'left':
                            bot.drive(-turn_speed, turn_speed)
                            cmd = "ROTATE-LEFT"
                        elif mp_pos == 'right':
                            bot.drive(turn_speed, -turn_speed)
                            cmd = "ROTATE-RIGHT"
                else:
                    if last_driving_command != "STOP":
                        bot.stop()
                    cmd = "STOP (NO TARGET)"
                
                active_drive_cmd = cmd
                if frame_count % 10 == 0:
                    print(f"🔍 Locked:{is_locked} | Dist:{int(mp_dist_cm) if mp_dist_cm else 0}cm | CMD:{cmd}")
                if cmd != last_driving_command:
                    last_driving_command = cmd
        
        # Visual overlay
        if follow_mode_active:
            cv2.putText(frame, "FOLLOW: ON", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"CMD: {active_drive_cmd}", (10, frame_height-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        
        if len(target_person_encodings) > 0:
            cv2.putText(frame, f"Enrolled: {target_person_name} ({len(target_person_encodings)} views)",
                       (frame_width-350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,136), 1)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ============================================
#   MAIN ROUTES
# ============================================

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


# ============================================
#   MOTOR / JOYSTICK CONTROL
# ============================================

@app.route('/manual_control', methods=['POST'])
def manual_control():
    """Handle joystick/manual controls: F, B, L, R, S"""
    data = request.json
    command = data.get('command', '').upper()
    
    if not bot:
        return jsonify({'error': 'Robot not connected'}), 500
    
    if command == 'F':
        bot.drive(MANUAL_SPEED, MANUAL_SPEED)
    elif command == 'B':
        bot.drive(-MANUAL_SPEED, -MANUAL_SPEED)
    elif command == 'L':
        bot.drive(-MANUAL_SPEED, MANUAL_SPEED)
    elif command == 'R':
        bot.drive(MANUAL_SPEED, -MANUAL_SPEED)
    elif command == 'S':
        bot.stop()
    else:
        return jsonify({'error': 'Invalid command'}), 400
    
    return jsonify({'status': 'ok', 'command': command})


# ============================================
#   VOICE CONTROL
# ============================================

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
            if llm:
                try:
                    response = llm.ask(text + " (answer in 20 words or less)")
                except:
                    response = "Sorry, I'm having trouble thinking right now."
            else:
                response = "AI is not available."
        
        print(f"🤖 {ROBOT_NAME}: {response}")
        
        if mouth:
            mouth.speak(response)
        
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


# ============================================
#   TEXT CHAT
# ============================================

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


# ============================================
#   SYSTEM STATUS
# ============================================

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


# ============================================
#   HAND CONTROLS
# ============================================

@app.route('/hand_control', methods=['POST'])
def hand_control():
    """Control robot hands (open/close)"""
    global active_arm_cmd
    if not bot:
        return jsonify({'error': 'Robot not connected'}), 500
    
    data = request.json
    command = data.get('command', '').upper()
    
    try:
        if command == 'LC':
            bot.open_left_hand()
            active_arm_cmd = "LEFT HAND - OPENED"
            return jsonify({'success': True, 'message': 'Left hand opened'})
        elif command == 'LO':
            bot.close_left_hand()
            active_arm_cmd = "LEFT HAND - CLOSED"
            return jsonify({'success': True, 'message': 'Left hand closed'})
        elif command == 'RC':
            bot.open_right_hand()
            active_arm_cmd = "RIGHT HAND - OPENED"
            return jsonify({'success': True, 'message': 'Right hand opened'})
        elif command == 'RO':
            bot.close_right_hand()
            active_arm_cmd = "RIGHT HAND - CLOSED"
            return jsonify({'success': True, 'message': 'Right hand closed'})
        else:
            return jsonify({'error': 'Invalid command'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/wrist_control', methods=['POST'])
def wrist_control():
    """Control robot wrists (0-180 degrees)"""
    global active_arm_cmd
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
            active_arm_cmd = f"LEFT WRIST - {angle}°"
            return jsonify({'success': True, 'message': f'Left wrist set to {angle}°'})
        elif side == 'R':
            bot.set_right_wrist(angle)
            active_arm_cmd = f"RIGHT WRIST - {angle}°"
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
    global active_arm_cmd
    if not bot:
        return jsonify({'error': 'Robot not connected'}), 500
    
    data = request.json
    command = data.get('command', '').upper()
    
    try:
        if command == 'BLU':
            bot.left_bicep_up()
            active_arm_cmd = "LEFT BICEP - UP"
            return jsonify({'success': True, 'message': 'Left bicep moving up'})
        elif command == 'BLD':
            bot.left_bicep_down()
            active_arm_cmd = "LEFT BICEP - DOWN"
            return jsonify({'success': True, 'message': 'Left bicep moving down'})
        elif command == 'BRU':
            bot.right_bicep_up()
            active_arm_cmd = "RIGHT BICEP - UP"
            return jsonify({'success': True, 'message': 'Right bicep moving up'})
        elif command == 'BRD':
            bot.right_bicep_down()
            active_arm_cmd = "RIGHT BICEP - DOWN"
            return jsonify({'success': True, 'message': 'Right bicep moving down'})
        elif command == 'BS':
            bot.stop_biceps()
            active_arm_cmd = "BICEPS STOPPED"
            return jsonify({'success': True, 'message': 'Biceps stopped'})
        else:
            return jsonify({'error': 'Invalid command'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================
#   PERSON-SPECIFIC TRACKING ENDPOINTS
# ============================================

def _verify_face_with_haar(image):
    """Verify face is present using OpenCV Haar cascade (fallback when face_recognition unavailable)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = fc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return len(faces) > 0, len(faces)

def _save_person_image(image, name, view):
    """Save person image to disk for enrollment (fallback mode)"""
    os.makedirs('person_images', exist_ok=True)
    filepath = os.path.join('person_images', f'{view}.jpg')
    cv2.imwrite(filepath, image)
    # Also save metadata
    import json
    meta_path = os.path.join('person_images', 'metadata.json')
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except:
            meta = {}
    meta['name'] = name
    meta['views'] = meta.get('views', [])
    if view not in meta['views']:
        meta['views'].append(view)
    with open(meta_path, 'w') as f:
        json.dump(meta, f)
    return filepath

@app.route('/upload_person_photo', methods=['POST'])
def upload_person_photo():
    """Upload front/back/side photos to train on person"""
    global target_person_encodings, target_person_name
    
    if 'photo' not in request.files:
        return jsonify({'error': 'No photo uploaded'}), 400
    
    file = request.files['photo']
    name = request.form.get('name', 'Target Person')
    view = request.form.get('view', 'front')
    
    try:
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Use face_recognition if available, otherwise use Haar cascade fallback
        if generate_face_encoding:
            encoding, error = generate_face_encoding(image)
            if error:
                return jsonify({'error': error}), 400
            target_person_encodings[view] = encoding
            target_person_name = name
            save_target_encoding(encoding, name, view)
        else:
            # Fallback: verify face with Haar cascade and save image
            has_face, face_count = _verify_face_with_haar(image)
            if not has_face:
                return jsonify({'error': 'No face detected. Please ensure your face is clearly visible.'}), 400
            if face_count > 1:
                return jsonify({'error': f'Multiple faces detected ({face_count}). Please ensure only one person is visible.'}), 400
            _save_person_image(image, name, view)
            target_person_name = name
            target_person_encodings[view] = True  # Mark as enrolled
        
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
    view = data.get('view', 'front')
    
    try:
        ret, frame = cap.read()
        if not ret:
            return jsonify({'error': 'Failed to capture frame'}), 500
        
        # Use face_recognition if available, otherwise use Haar cascade fallback
        if generate_face_encoding:
            encoding, error = generate_face_encoding(frame)
            if error:
                return jsonify({'error': error}), 400
            target_person_encodings[view] = encoding
            target_person_name = name
            save_target_encoding(encoding, name, view)
        else:
            # Fallback: verify face with Haar cascade and save image
            has_face, face_count = _verify_face_with_haar(frame)
            if not has_face:
                return jsonify({'error': 'No face detected. Please position yourself in front of the camera.'}), 400
            if face_count > 1:
                return jsonify({'error': f'Multiple faces detected ({face_count}). Please ensure only one person is visible.'}), 400
            _save_person_image(frame, name, view)
            target_person_name = name
            target_person_encodings[view] = True  # Mark as enrolled
        
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
#   MODEL TRAINING ENDPOINTS
# ============================================

@app.route('/train_model', methods=['POST'])
def train_model():
    """Add a new Q&A training pair"""
    if not llm:
        return jsonify({'error': 'LLM not available'}), 500
    
    data = request.json
    question = data.get('question', '').strip()
    answer = data.get('answer', '').strip()
    
    if not question or not answer:
        return jsonify({'error': 'Both question and answer are required'}), 400
    
    try:
        llm.add_training_pair(question, answer)
        return jsonify({
            'success': True,
            'message': f'Training pair added successfully!',
            'total_pairs': len(llm.get_training_pairs())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/training_data', methods=['GET'])
def get_training_data():
    """Get all training pairs"""
    if not llm:
        return jsonify({'error': 'LLM not available'}), 500
    
    return jsonify({
        'pairs': llm.get_training_pairs(),
        'total': len(llm.get_training_pairs())
    })


@app.route('/training_data/<int:index>', methods=['DELETE'])
def delete_training_data(index):
    """Delete a training pair by index"""
    if not llm:
        return jsonify({'error': 'LLM not available'}), 500
    
    try:
        if llm.remove_training_pair(index):
            return jsonify({
                'success': True,
                'message': 'Training pair removed',
                'total_pairs': len(llm.get_training_pairs())
            })
        else:
            return jsonify({'error': 'Invalid index'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/reload_training', methods=['POST'])
def reload_training():
    """Reload training data from file"""
    if not llm:
        return jsonify({'error': 'LLM not available'}), 500
    
    try:
        llm.reload_training_data()
        return jsonify({
            'success': True,
            'message': 'Training data reloaded',
            'total_pairs': len(llm.get_training_pairs())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================
#   MAIN
# ============================================

if __name__ == '__main__':
    from config import WEB_PORT, WEB_HOST
    print(f"\n{'='*60}")
    print(f"🤖 {ROBOT_NAME} Complete Control Interface")
    print(f"{'='*60}")
    print(f"\n📱 Access: http://{RASPBERRY_PI_IP}:{WEB_PORT}")
    print(f"\n✨ Features:")
    print(f"   📹 Live Camera Feed")
    print(f"   🎤 Voice Control")
    print(f"   🕹️  Joystick Controls")
    print(f"   👤 Face Detection & Person Training")
    print(f"   🎯 Follow Mode")
    print(f"   🖐️  Hand & Arm Controls")
    print(f"   🧠 AI Model Training")
    print(f"\n{'='*60}\n")
    
    try:
        import os
        if os.path.exists('cert.pem') and os.path.exists('key.pem'):
            print(f"🔒 Starting server with HTTPS (https://{RASPBERRY_PI_IP}:{WEB_PORT})")
            app.run(host=WEB_HOST, port=WEB_PORT, debug=False, threaded=True, ssl_context=('cert.pem', 'key.pem'))
        else:
            print(f"🔓 Starting server with HTTP (http://{RASPBERRY_PI_IP}:{WEB_PORT})")
            app.run(host=WEB_HOST, port=WEB_PORT, debug=False, threaded=True)
    finally:
        if bot:
            bot.close()
        if cap:
            cap.release()
