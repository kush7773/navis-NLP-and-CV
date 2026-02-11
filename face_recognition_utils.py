"""
Face Recognition Helper Functions
For person-specific tracking
"""

import face_recognition
import pickle
import os
from datetime import datetime
import cv2
import numpy as np


def save_target_encoding(encoding, name="Target Person", view="front"):
    """
    Save face encoding to disk
    Supports multiple views (front, back, side) for robust tracking
    """
    # Load existing data if available
    data = {}
    if os.path.exists('target_person.pkl'):
        try:
            with open('target_person.pkl', 'rb') as f:
                data = pickle.load(f)
        except:
            data = {}
    
    # Initialize structure if needed
    if 'encodings' not in data:
        data['encodings'] = {}
    
    # Add new encoding
    data['encodings'][view] = encoding
    data['name'] = name
    data['timestamp'] = datetime.now().isoformat()
    data['views'] = list(data['encodings'].keys())
    
    with open('target_person.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✅ Saved {view} encoding for: {name} (Total views: {len(data['encodings'])})")


def load_target_encoding():
    """
    Load saved face encodings (supports multiple views)
    Returns: (encodings_dict, name) where encodings_dict = {'front': encoding, 'back': encoding, ...}
    """
    if os.path.exists('target_person.pkl'):
        try:
            with open('target_person.pkl', 'rb') as f:
                data = pickle.load(f)
                
                # Handle old format (single encoding)
                if 'encoding' in data:
                    print(f"✅ Loaded encoding for: {data['name']} (legacy format)")
                    return {'front': data['encoding']}, data['name']
                
                # New format (multiple encodings)
                if 'encodings' in data:
                    print(f"✅ Loaded {len(data['encodings'])} encodings for: {data['name']}")
                    print(f"   Views: {', '.join(data['views'])}")
                    return data['encodings'], data['name']
                    
        except Exception as e:
            print(f"❌ Error loading encoding: {e}")
            return {}, None
    return {}, None


def generate_face_encoding(image):
    """
    Generate face encoding from image
    Returns: (encoding, error_message)
    """
    try:
        # Convert BGR to RGB (OpenCV uses BGR, face_recognition uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image, model='hog')
        
        if len(face_locations) == 0:
            return None, "No face detected. Please ensure your face is clearly visible."
        
        if len(face_locations) > 1:
            return None, f"Multiple faces detected ({len(face_locations)}). Please ensure only one person is visible."
        
        # Generate encoding
        encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        if len(encodings) == 0:
            return None, "Could not generate face encoding. Please try again with better lighting."
        
        print(f"✅ Generated face encoding successfully")
        return encodings[0], None
        
    except Exception as e:
        return None, f"Error generating encoding: {str(e)}"


def match_target_person(face_encoding, target_encodings_dict, threshold=0.6):
    """
    Check if detected face matches target person (any view)
    target_encodings_dict: {'front': encoding, 'back': encoding, ...}
    Returns: (is_match, confidence, matched_view)
    """
    if not target_encodings_dict or len(target_encodings_dict) == 0:
        return False, 0.0, None
    
    try:
        best_match = False
        best_confidence = 0.0
        best_view = None
        
        # Compare against all saved views
        for view, target_encoding in target_encodings_dict.items():
            # Compare faces (returns distance, lower = more similar)
            distance = face_recognition.face_distance([target_encoding], face_encoding)[0]
            
            # Convert distance to confidence (0-1, higher = more confident)
            confidence = 1 - distance
            
            # Check if this is the best match so far
            if confidence > best_confidence:
                best_confidence = confidence
                best_view = view
                best_match = distance < threshold
        
        return best_match, best_confidence, best_view
        
    except Exception as e:
        print(f"❌ Error matching face: {e}")
        return False, 0.0, None


def detect_and_match_faces(frame, target_encodings_dict, threshold=0.6):
    """
    Detect all faces in frame and match against target (all views)
    Returns: list of (bbox, is_target, confidence, matched_view)
    bbox format: (x, y, w, h)
    """
    results = []
    
    try:
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces (returns in (top, right, bottom, left) format)
        face_locations = face_recognition.face_locations(rgb_frame, model='hog')
        
        if len(face_locations) == 0:
            return results
        
        # Generate encodings for all detected faces
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Match each face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Convert to (x, y, w, h) format
            x = left
            y = top
            w = right - left
            h = bottom - top
            
            # Match against target (all views)
            is_match, confidence, matched_view = match_target_person(
                face_encoding, target_encodings_dict, threshold
            )
            
            results.append({
                'bbox': (x, y, w, h),
                'is_target': is_match,
                'confidence': confidence,
                'matched_view': matched_view
            })
        
        return results
        
    except Exception as e:
        print(f"❌ Error detecting faces: {e}")
        return results
