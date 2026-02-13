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

print("\n📸 Testing Webcam Match...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("Please look at the camera...")
time.sleep(1)

ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Failed to capture frame")
    exit()

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
locs = face_recognition.face_locations(rgb, model='hog')
print(f"faces found: {len(locs)}")

if len(locs) > 0:
    encs = face_recognition.face_encodings(rgb, locs)
    if len(encs) > 0:
        test_enc = encs[0]
        # Match
        for view, target_enc in data['encodings'].items():
            # Force convert for test
            if isinstance(target_enc, list) or isinstance(target_enc, tuple):
                target_enc = np.array(target_enc)
            
            dist = face_recognition.face_distance([target_enc], test_enc)[0]
            print(f"  - Match vs {view}: Dist={dist:.4f} (Threshold 0.6)")
            if dist < 0.6:
                print("    ✅ MATCH!")
            else:
                print("    ❌ NO MATCH")
    else:
        print("❌ Face detected but no encoding generated")
else:
    print("❌ No face detected in live frame (lighting?)")
