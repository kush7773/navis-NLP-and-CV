
# ============================================
#   PERSON-SPECIFIC TRACKING ENDPOINTS
# ============================================

@app.route('/upload_person_photo', methods=['POST'])
def upload_person_photo():
    """Upload front/back photos to train on person"""
    global target_person_encoding, target_person_name
    
    if 'photo' not in request.files:
        return jsonify({'error': 'No photo uploaded'}), 400
    
    file = request.files['photo']
    name = request.form.get('name', 'Target Person')
    
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
        
        # Save encoding
        target_person_encoding = encoding
        target_person_name = name
        save_target_encoding(encoding, name)
        
        return jsonify({
            'success': True,
            'message': f'Successfully enrolled {name}!'
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/capture_person', methods=['POST'])
def capture_person():
    """Capture current frame and train on person"""
    global target_person_encoding, target_person_name, cap
    
    if not cap or not cap.isOpened():
        return jsonify({'error': 'Camera not available'}), 500
    
    name = request.json.get('name', 'Target Person') if request.json else 'Target Person'
    
    try:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            return jsonify({'error': 'Failed to capture frame'}), 500
        
        # Generate encoding
        encoding, error = generate_face_encoding(frame)
        
        if error:
            return jsonify({'error': error}), 400
        
        # Save encoding
        target_person_encoding = encoding
        target_person_name = name
        save_target_encoding(encoding, name)
        
        return jsonify({
            'success': True,
            'message': f'I will remember you, {name}!'
        })
        
    except Exception as e:
        return jsonify({'error': f'Capture failed: {str(e)}'}), 500


@app.route('/clear_person', methods=['POST'])
def clear_person():
    """Clear enrolled person"""
    global target_person_encoding, target_person_name
    
    try:
        target_person_encoding = None
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
        'enrolled': target_person_encoding is not None,
        'name': target_person_name if target_person_encoding else None
    })
