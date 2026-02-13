import pickle
import numpy as np
import os
import face_recognition
import cv2
import time

print("🔍 Analyzing Face Training Data...")

if not os.path.exists('target_person.pkl'):
    print("❌ No 'target_person.pkl' found!")
    exit()

try:
    with open('target_person.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ Loaded pickle file. Keys: {list(data.keys())}")
    
    if 'encodings' in data:
        for view, enc in data['encodings'].items():
            print(f"  - View: {view}")
            print(f"    Type: {type(enc)}")
            if isinstance(enc, np.ndarray):
                print(f"    Shape: {enc.shape}")
                print(f"    Dtype: {enc.dtype}")
            elif isinstance(enc, list):
                print(f"    Length: {len(enc)} (List)")
            elif isinstance(enc, tuple):
                print(f"    Length: {len(enc)} (Tuple)")
            else:
                print(f"    Unknown type: {type(enc)}")
    else:
        print("❌ No 'encodings' key in data!")

except Exception as e:
    print(f"❌ Error loading pickle: {e}")
    exit()

print("📷 Testing Webcam Match (Loop)... Press CTRL+C to stop")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5) # Speed up
        
        locs = face_recognition.face_locations(small_frame, model='hog')
        print(f"\rFound {len(locs)} faces...", end="")
        
        if len(locs) > 0:
            encs = face_recognition.face_encodings(small_frame, locs)
            if len(encs) > 0:
                test_enc = encs[0]
                # Match
                for view, target_enc in data['encodings'].items():
                    # Force convert
                    if isinstance(target_enc, (list, tuple)):
                        target_enc = np.array(target_enc)
                    
                    dist = face_recognition.face_distance([target_enc], test_enc)[0]
                    if dist < 0.6:
                        print(f" ✅ MATCH {view.upper()} (Dist: {dist:.3f})   ", end="")
                    else:
                        print(f" ❌ NO MATCH (Dist: {dist:.3f})   ", end="")
        
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopped.")

cap.release()
