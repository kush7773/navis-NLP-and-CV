import cv2
import numpy as np
import os
import time
from flask import Flask, render_template_string, Response, redirect, url_for
from serial_bridge import RobotBridge

# --- CONFIGURATION ---
AUTO_SPEED = 90       # Speed for Face Tracking
MANUAL_SPEED = 150    # Speed for Web Buttons (Faster)
TURN_SPEED = 110
DEAD_ZONE = 60
TRAINER_FILE = 'trainer.yml'

# --- HARDWARE ---
try:
    port = '/dev/ttyUSB0'
    if os.path.exists('/dev/ttyACM0'): port = '/dev/ttyACM0'
    bot = RobotBridge(port=port, baud_rate=115200)
    print(f"? ESP32 Connected: {port}")
except:
    print("?? ESP32 Not Found")
    bot = None

# --- VISION ---
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Global State
tracking_active = False  # True = AI Driving
model_trained = False
current_frame = None

# Load model if exists
if os.path.exists(TRAINER_FILE):
    try:
        recognizer.read(TRAINER_FILE)
        model_trained = True
    except: pass

app = Flask(__name__)

# --- WEBSITE (D-Pad + AI Controls) ---
HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NAVIS ULTIMATE</title>
    <style>
        body { font-family: 'Courier New', sans-serif; background: #0d0d0d; color: #00ffcc; text-align: center; margin: 0; padding: 10px; }
        h1 { margin: 5px 0; font-size: 20px; text-shadow: 0 0 10px #00ffcc; }
        
        /* Video Feed */
        .video-box { border: 2px solid #00ffcc; border-radius: 10px; overflow: hidden; max-width: 640px; margin: 0 auto; box-shadow: 0 0 20px rgba(0, 255, 204, 0.2); }
        img { width: 100%; display: block; }

        /* Manual D-Pad */
        .d-pad { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; max-width: 300px; margin: 20px auto; }
        .btn-m { padding: 20px; font-size: 24px; border: 2px solid #00ffcc; background: rgba(0,0,0,0.5); color: #00ffcc; border-radius: 10px; cursor: pointer; touch-action: manipulation; }
        .btn-m:active { background: #00ffcc; color: black; }
        
        /* AI Controls */
        .ai-controls { margin-top: 20px; border-top: 1px solid #333; padding-top: 15px; }
        .btn-ai { padding: 12px 25px; margin: 5px; font-size: 16px; border: none; border-radius: 50px; cursor: pointer; font-weight: bold; }
        .btn-lock { background: #ff00cc; color: white; }
        .btn-auto { background: #00ffcc; color: black; }
        
        /* Stop Button */
        .btn-stop { width: 90%; max-width: 300px; padding: 15px; background: #ff3333; color: white; font-size: 20px; border: none; border-radius: 10px; margin-top: 20px; font-weight: bold; box-shadow: 0 0 15px #ff0000; }
    </style>
</head>
<body>
    <h1>? NAVIS COMMAND</h1>
    
    <div class="video-box">
        <img src="{{ url_for('video_feed') }}">
    </div>
    
    <div style="margin: 5px; color: #aaa;">STATUS: {{ message }}</div>

    <div class="d-pad">
        <div></div>
        <a href="/manual/fwd"><button class="btn-m">??</button></a>
        <div></div>
        
        <a href="/manual/left"><button class="btn-m">??</button></a>
        <a href="/manual/stop"><button class="btn-m" style="font-size:16px;">?</button></a>
        <a href="/manual/right"><button class="btn-m">??</button></a>
        
        <div></div>
        <a href="/manual/back"><button class="btn-m">??</button></a>
        <div></div>
    </div>

    <a href="/manual/stop"><button class="btn-stop">?? EMERGENCY STOP ??</button></a>

    <div class="ai-controls">
        <h3>? AI PILOT</h3>
        <a href="/train_now"><button class="btn-ai btn-lock">? LEARN FACE</button></a>
        <a href="/start_auto"><button class="btn-ai btn-auto">? AUTO FOLLOW</button></a>
    </div>

</body>
</html>
"""

@app.route('/')
def index():
    status = "MANUAL MODE"
    if tracking_active: status = "? AI DRIVING..."
    return render_template_string(HTML_PAGE, message=status)

# --- MANUAL CONTROL ROUTES ---
@app.route('/manual/<cmd>')
def manual_control(cmd):
    global tracking_active
    
    # 1. Disable AI Tracking (Safety First)
    tracking_active = False 
    
    # 2. Execute Command
    if bot:
        if cmd == 'fwd':   bot.drive(MANUAL_SPEED, MANUAL_SPEED)
        elif cmd == 'back': bot.drive(-MANUAL_SPEED, -MANUAL_SPEED)
        elif cmd == 'left': bot.drive(-MANUAL_SPEED, MANUAL_SPEED)
        elif cmd == 'right': bot.drive(MANUAL_SPEED, -MANUAL_SPEED)
        elif cmd == 'stop': bot.stop()
        
    # 3. Stay on page (204 = No Content Refresh)
    return ('', 204)

# --- AI TRAINING & CONTROL ---
@app.route('/train_now')
def train_now():
    global model_trained, current_frame
    if current_frame is None: return redirect(url_for('index'))

    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        
        data, labels = [], []
        for i in range(30):
            data.append(face_roi)
            labels.append(1)

        recognizer.train(data, np.array(labels))
        recognizer.save(TRAINER_FILE)
        model_trained = True
    
    return redirect(url_for('index'))

@app.route('/start_auto')
def start_auto():
    global tracking_active
    if model_trained: tracking_active = True
    return redirect(url_for('index'))

# --- VIDEO LOOP ---
def generate_frames():
    global current_frame
    while True:
        success, frame = cap.read()
        if not success: break
        current_frame = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        
        left, right = 0, 0
        status = ""
        
        # Draw Faces
        for (x, y, w, h) in faces:
            color = (0, 0, 255)
            if model_trained:
                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                if confidence < 80:
                    status = "LOCKED"
                    color = (0, 255, 0)
                    
                    # ONLY calculate moves if AI is active
                    if tracking_active:
                        cx = x + w // 2
                        error = cx - 320
                        if abs(error) > DEAD_ZONE:
                            if error > 0: # Right
                                left, right = TURN_SPEED, -TURN_SPEED
                            else: # Left
                                left, right = -TURN_SPEED, TURN_SPEED
                        else:
                            if w < 100: left, right = AUTO_SPEED, AUTO_SPEED
                            elif w > 180: left, right = -AUTO_SPEED, -AUTO_SPEED
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Drive Motors (Only if AI is Active)
        if tracking_active and bot:
            bot.drive(left, right)
        # Note: If tracking is False, we DO NOT send stop() here.
        # This allows the Manual Buttons to control the motors without interference.

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        if bot: bot.close()
        cap.release()
