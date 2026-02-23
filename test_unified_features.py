import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import json

# Mock hardware modules BEFORE importing the app
sys.modules['serial_bridge'] = MagicMock()
sys.modules['tts'] = MagicMock()
sys.modules['llm_handler_updated'] = MagicMock()
sys.modules['speech_recognition'] = MagicMock()
sys.modules['pydub'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['face_recognition'] = MagicMock()
sys.modules['face_recognition_utils'] = MagicMock()

# Mock specific attributes
sys.modules['face_recognition_utils'].load_target_encoding.return_value = ({}, "Unknown")
sys.modules['cv2'].VideoCapture.return_value.isOpened.return_value = True
sys.modules['cv2'].VideoCapture.return_value.read.return_value = (True, MagicMock())
sys.modules['cv2'].data.haarcascades = "mock_path/"

# Import the app
from navis_complete_control import app, check_follow_command

class TestNavisUnified(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_status_endpoint(self):
        """Test status endpoint"""
        response = self.app.get('/status')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('robot', data)
        self.assertIn('follow_active', data)

    def test_manual_control(self):
        """Test joystick controls"""
        commands = ['F', 'B', 'L', 'R', 'S']
        for cmd in commands:
            response = self.app.post('/manual_control', 
                                   json={'command': cmd})
            self.assertEqual(response.status_code, 200)

    def test_follow_command_logic(self):
        """Test text command parsing"""
        self.assertEqual(check_follow_command("Follow me Navis please"), "FOLLOW")
        self.assertEqual(check_follow_command("Stop Navis now"), "STOP")
        self.assertIsNone(check_follow_command("Hello robot"))

    def test_toggle_face_detection(self):
        """Test face detection toggle"""
        response = self.app.post('/toggle_face_detection')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('enabled', data)

    def test_hand_control(self):
        """Test hand control endpoint"""
        response = self.app.post('/hand_control', json={'command': 'LC'})
        self.assertEqual(response.status_code, 200)

    def test_wrist_control(self):
        """Test wrist control endpoint"""
        response = self.app.post('/wrist_control', json={'side': 'L', 'angle': 90})
        self.assertEqual(response.status_code, 200)

    def test_bicep_control(self):
        """Test bicep control endpoint"""
        response = self.app.post('/bicep_control', json={'command': 'BLU'})
        self.assertEqual(response.status_code, 200)

    def test_person_training_endpoints(self):
        """Test person training endpoints"""
        # Status
        response = self.app.get('/person_status')
        self.assertEqual(response.status_code, 200)
        
        # Clear
        response = self.app.post('/clear_person')
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
