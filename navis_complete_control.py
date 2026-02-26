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
    TURN_SPEED,
    PERSISTENCE_FRAMES,
    SMOOTHING_ALPHA,
    DEAD_ZONE,
    HYSTERESIS_ZONE,
    MIN_FRAMES_BETWEEN_DIR_CHANGE
)

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

# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
    """Generate video frames with face detection and follow mode"""
    global show_face_detection, follow_mode_active, target_person_encodings
    global active_drive_cmd, active_arm_cmd
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
    upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
    
    frame_count = 0
    process_every_n_frames = 3
    
    # Cache variables for skipped frames
    cached_faces = []
    cached_humans = []
    cached_target_found = False
    cached_target_x = 0
    cached_target_w = 0
    cached_detected_human = False
    
    # === Smoothing, Persistence, Hysteresis ===
    smoothed_x = 0.0            # EMA-smoothed target X position
    smoothed_w = 0.0            # EMA-smoothed target width
    frames_since_seen = 999     # How many frames since we last saw the target
    last_known_x = 0            # Last known good X position
    last_known_w = 0            # Last known good width
    current_turn_dir = 0        # -1=LEFT, 0=STOPPED, 1=RIGHT (for hysteresis)
    frames_since_dir_change = 0 # Rate limiter: frames since last motor direction change
    last_motor_cmd = ""         # Tracks last command to avoid redundant serial writes
    
    # Distance Heuristics
    KNOWN_FACE_WIDTH_CM = 15.0
    FOCAL_LENGTH = 400.0
    
    while True:
        if not cap:
            break
            
        ret, frame = cap.read()
        if not ret:
            break
        
        # Fix inverted camera — flip horizontally (mirror)
        frame = cv2.flip(frame, 1)
        
        
        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width // 2
        
        distance_status = ""
        distance_color = (255, 255, 255)
        
        # Only process every N frames
        if frame_count % process_every_n_frames == 0:
            cached_detected_human = False
            cached_target_x = frame_center_x
            cached_target_w = 0
            cached_faces = []
            cached_humans = []
            cached_target_found = False
            
            # Create a downscaled frame for faster processing
            scale_factor = 0.25  # Process at 1/4 resolution
            small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            
            if show_face_detection or follow_mode_active:
                
                if len(target_person_encodings) > 0:
                    face_results = detect_and_match_faces(small_frame, target_person_encodings, match_threshold)
                    
                    for result in face_results:
                        # Scale bounding boxes back up
                        x, y, w, h = [int(v / scale_factor) for v in result['bbox']]
                        is_target = result['is_target']
                        confidence = result['confidence']
                        matched_view = result['matched_view']
                        
                        cached_faces.append((x, y, w, h, is_target, confidence, matched_view))
                        
                        if is_target:
                            cached_target_x = x + w // 2
                            cached_target_w = w
                            
                            # Estimate Distance
                            if w > 0:
                                distance_cm = (KNOWN_FACE_WIDTH_CM * FOCAL_LENGTH) / w
                            else:
                                distance_cm = 0.0
                                
                            cached_target_found = True
                            cached_detected_human = True
                    
                    if not cached_target_found:
                        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                        
                        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
                        upper_bodies = upper_body_cascade.detectMultiScale(gray, 1.1, 3)
                        full_bodies = body_cascade.detectMultiScale(gray, 1.1, 3)
                        
                        humans = []
                        if len(faces) > 0:
                            humans = [(int(x/scale_factor), int(y/scale_factor), int(w/scale_factor), int(h/scale_factor), "FACE") for (x, y, w, h) in faces]
                        elif len(upper_bodies) > 0:
                            humans = [(int(x/scale_factor), int(y/scale_factor), int(w/scale_factor), int(h/scale_factor), "UPPER BODY") for (x, y, w, h) in upper_bodies]
                        elif len(full_bodies) > 0:
                            humans = [(int(x/scale_factor), int(y/scale_factor), int(w/scale_factor), int(h/scale_factor), "FULL BODY") for (x, y, w, h) in full_bodies]
                        
                        cached_humans = humans
                        for (x, y, w, h, label) in humans:
                            if w > cached_target_w:
                                cached_target_x = x + w // 2
                                cached_target_w = w
                                cached_detected_human = True
                
                else:
                    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                    
                    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
                    upper_bodies = upper_body_cascade.detectMultiScale(gray, 1.1, 3)
                    full_bodies = body_cascade.detectMultiScale(gray, 1.1, 3)
                    
                    humans = []
                    if len(faces) > 0:
                        humans = [(int(x/scale_factor), int(y/scale_factor), int(w/scale_factor), int(h/scale_factor), "FACE") for (x, y, w, h) in faces]
                    elif len(upper_bodies) > 0:
                        humans = [(int(x/scale_factor), int(y/scale_factor), int(w/scale_factor), int(h/scale_factor), "UPPER BODY") for (x, y, w, h) in upper_bodies]
                    elif len(full_bodies) > 0:
                        humans = [(int(x/scale_factor), int(y/scale_factor), int(w/scale_factor), int(h/scale_factor), "FULL BODY") for (x, y, w, h) in full_bodies]
                    
                    cached_humans = humans
                    for (x, y, w, h, label) in humans:
                        cached_detected_human = True
                        if w > cached_target_w:
                            cached_target_x = x + w // 2
                            cached_target_w = w
        
        frame_count += 1
        
        # === PERSISTENCE BUFFER & EMA SMOOTHING ===
        if cached_detected_human:
            # Target is visible — update smoothed values and reset persistence counter
            frames_since_seen = 0
            last_known_x = cached_target_x
            last_known_w = cached_target_w
            smoothed_x = SMOOTHING_ALPHA * cached_target_x + (1 - SMOOTHING_ALPHA) * smoothed_x
            smoothed_w = SMOOTHING_ALPHA * cached_target_w + (1 - SMOOTHING_ALPHA) * smoothed_w
        else:
            frames_since_seen += 1
            # Decay smoothed values toward center during persistence window
            if frames_since_seen <= PERSISTENCE_FRAMES:
                # Slowly drift toward last known position (don't snap to center)
                smoothed_x = 0.95 * smoothed_x + 0.05 * last_known_x
                smoothed_w = smoothed_w  # Hold width steady
        
        # Decide if we should "act" on detection (real detection OR within persistence window)
        effectively_detected = cached_detected_human or (frames_since_seen <= PERSISTENCE_FRAMES)
        
        # Use smoothed values for motor control
        target_x = int(smoothed_x)
        target_w = int(smoothed_w)
        
        # Draw bounding boxes from cache
        if show_face_detection or follow_mode_active:
            for (x, y, w, h, is_target, confidence, matched_view) in cached_faces:
                if is_target:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    
                    # Calculate estimated distance for rendering
                    distance_cm = (KNOWN_FACE_WIDTH_CM * FOCAL_LENGTH) / w if w > 0 else 0
                    
                    # Append raw pixel widths to allow visual calibration
                    label = f"TARGET | Dist: {distance_cm:.1f}cm | W: {w}px"
                    if matched_view:
                        label += f" ({matched_view.upper()})"
                    cv2.putText(frame, label, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, f"OTHER ({confidence:.2f})", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                               
            for (x, y, w, h, label) in cached_humans:
                if len(target_person_encodings) > 0 and not cached_target_found:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    cv2.putText(frame, f"{label} (TRACKING)", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                elif len(target_person_encodings) == 0:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Follow mode motor control (with hysteresis + rate limiting)
        frames_since_dir_change += 1
        
        if follow_mode_active and effectively_detected and bot:
            error = target_x - frame_center_x
            
            # === PROPORTIONAL SPEED CONTROL ===
            max_error = frame_width // 2
            error_ratio = min(abs(error) / max(max_error, 1), 1.0)
            # Minimum turn is 25% of TURN_SPEED (gentler minimum)
            proportional_turn = int(TURN_SPEED * 0.25 + TURN_SPEED * 0.75 * error_ratio)
            
            OPTIMAL_MIN_SIZE = 80
            OPTIMAL_MAX_SIZE = 150
            STOP_DISTANCE = 200
            
            if target_w < OPTIMAL_MIN_SIZE:
                distance_ratio = max(0.0, 1.0 - (target_w / max(OPTIMAL_MIN_SIZE, 1)))
                proportional_forward = int(AUTO_SPEED * 0.4 + AUTO_SPEED * 0.6 * distance_ratio)
            else:
                proportional_forward = int(AUTO_SPEED * 0.3)
            
            # === HYSTERESIS LOGIC ===
            # Determine desired turn direction with hysteresis
            if current_turn_dir == 0:
                # Currently stopped — only START turning if error exceeds DEAD_ZONE
                if error > DEAD_ZONE:
                    desired_dir = 1   # RIGHT
                elif error < -DEAD_ZONE:
                    desired_dir = -1  # LEFT
                else:
                    desired_dir = 0   # CENTERED
            else:
                # Currently turning — only STOP if error drops below HYSTERESIS_ZONE
                if abs(error) < HYSTERESIS_ZONE:
                    desired_dir = 0   # Close enough, stop turning
                elif error > 0:
                    desired_dir = 1
                else:
                    desired_dir = -1
            
            # === RATE LIMITER ===
            # Block direction changes if too soon (prevents rapid flip-flop)
            if desired_dir != current_turn_dir:
                if frames_since_dir_change < MIN_FRAMES_BETWEEN_DIR_CHANGE:
                    desired_dir = current_turn_dir  # Force keep current direction
                else:
                    current_turn_dir = desired_dir
                    frames_since_dir_change = 0
            
            # === EXECUTE MOTOR COMMAND ===
            new_cmd = ""
            if target_w > STOP_DISTANCE:
                new_cmd = "STOP_CLOSE"
                if new_cmd != last_motor_cmd:
                    bot.stop()
                active_drive_cmd = "STOP (TOO CLOSE)"
                distance_status = "TOO CLOSE - STOPPED"
                distance_color = (0, 0, 255)
                current_turn_dir = 0
                
            elif target_w > OPTIMAL_MAX_SIZE:
                # Close enough — just orient
                if desired_dir == 0:
                    new_cmd = "STOP_OPTIMAL"
                    if new_cmd != last_motor_cmd:
                        bot.stop()
                    active_drive_cmd = "STOP (OPTIMAL)"
                    distance_status = "OPTIMAL - CENTERED"
                    distance_color = (0, 255, 0)
                elif desired_dir == 1:
                    new_cmd = f"TURN_R_{proportional_turn}"
                    if new_cmd != last_motor_cmd:
                        bot.drive(-proportional_turn, proportional_turn)
                    active_drive_cmd = "TURNING RIGHT"
                    distance_status = "TURNING RIGHT"
                    distance_color = (255, 255, 0)
                else:
                    new_cmd = f"TURN_L_{proportional_turn}"
                    if new_cmd != last_motor_cmd:
                        bot.drive(proportional_turn, -proportional_turn)
                    active_drive_cmd = "TURNING LEFT"
                    distance_status = "TURNING LEFT"
                    distance_color = (255, 255, 0)
                    
            elif target_w < OPTIMAL_MIN_SIZE:
                # Far away — drive forward with steering
                if desired_dir == 0:
                    new_cmd = f"FWD_{proportional_forward}"
                    if new_cmd != last_motor_cmd:
                        bot.drive(proportional_forward, proportional_forward)
                    active_drive_cmd = "MOVING FORWARD"
                    distance_status = "MOVING FORWARD"
                    distance_color = (0, 255, 255)
                elif desired_dir == 1:
                    new_cmd = f"FWD_R_{proportional_forward}"
                    if new_cmd != last_motor_cmd:
                        bot.drive(proportional_forward, proportional_forward // 2)
                    active_drive_cmd = "FORWARD + RIGHT"
                    distance_status = "FORWARD + RIGHT"
                    distance_color = (0, 255, 255)
                else:
                    new_cmd = f"FWD_L_{proportional_forward}"
                    if new_cmd != last_motor_cmd:
                        bot.drive(proportional_forward // 2, proportional_forward)
                    active_drive_cmd = "FORWARD + LEFT"
                    distance_status = "FORWARD + LEFT"
                    distance_color = (0, 255, 255)
            else:
                # Optimal distance zone
                if desired_dir == 0:
                    new_cmd = "STOP_ZONE"
                    if new_cmd != last_motor_cmd:
                        bot.stop()
                    active_drive_cmd = "STOP (OPTIMAL)"
                    distance_status = "OPTIMAL - CENTERED"
                    distance_color = (0, 255, 0)
                elif desired_dir == 1:
                    new_cmd = f"ADJ_R_{proportional_turn}"
                    if new_cmd != last_motor_cmd:
                        bot.drive(-proportional_turn, proportional_turn)
                    active_drive_cmd = "ADJUSTING RIGHT"
                    distance_status = "ADJUSTING RIGHT"
                    distance_color = (255, 255, 0)
                else:
                    new_cmd = f"ADJ_L_{proportional_turn}"
                    if new_cmd != last_motor_cmd:
                        bot.drive(proportional_turn, -proportional_turn)
                    active_drive_cmd = "ADJUSTING LEFT"
                    distance_status = "ADJUSTING LEFT"
                    distance_color = (255, 255, 0)
            
            last_motor_cmd = new_cmd
        
        elif follow_mode_active and not effectively_detected and bot:
            if last_motor_cmd != "SEARCH_STOP":
                bot.stop()
                last_motor_cmd = "SEARCH_STOP"
            current_turn_dir = 0
            distance_status = "SEARCHING..."
            distance_color = (255, 165, 0)
        
        # Visual overlay
        if follow_mode_active:
            cv2.putText(frame, "FOLLOW MODE: ON", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if effectively_detected and distance_status:
                cv2.putText(frame, distance_status, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, distance_color, 2)
                distance_text = f"Distance: {target_w}px"
                cv2.putText(frame, distance_text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                # Show persistence indicator when coasting on memory
                if not cached_detected_human and frames_since_seen > 0:
                    persist_text = f"MEMORY ({PERSISTENCE_FRAMES - frames_since_seen} frames left)"
                    cv2.putText(frame, persist_text, (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            else:
                cv2.putText(frame, "SEARCHING...", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        
        if len(target_person_encodings) > 0:
            enrolled_text = f"Enrolled: {target_person_name} ({len(target_person_encodings)} views)"
            cv2.putText(frame, enrolled_text, (10, frame_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 136), 1)
        
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
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
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
