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
from config import (
    VOICE_PORT, 
    ROBOT_NAME, 
    FOLLOW_COMMANDS, 
    STOP_COMMANDS,
    RASPBERRY_PI_IP,
    SERIAL_PORT,
    SERIAL_BAUD,
    MANUAL_SPEED
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
    """Generate camera frames with optional face detection"""
    global show_face_detection, follow_mode_active
    
    while True:
        if not cap:
            break
            
        success, frame = cap.read()
        if not success:
            break
        
        # Apply face detection if enabled
        if show_face_detection:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)
            
            for (x, y, w, h) in faces:
                color = (0, 255, 0) if follow_mode_active else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Add label
                label = "TRACKING" if follow_mode_active else "DETECTED"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add status overlay
        status_text = "FOLLOW MODE: ON" if follow_mode_active else "MANUAL CONTROL"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if follow_mode_active else (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

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
            # Normal AI conversation
            if llm:
                response = llm.ask(text)
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
