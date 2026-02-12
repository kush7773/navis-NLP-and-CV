#!/usr/bin/env python3
"""
Quick test script for person tracking endpoints
Run this AFTER starting web_voice_interface.py
"""

import requests
import json

# Update this to your Raspberry Pi IP
BASE_URL = "http://192.168.0.112:5000"

def test_person_status():
    """Test /person_status endpoint"""
    print("\n🔍 Testing /person_status endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/person_status", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_capture_person():
    """Test /capture_person endpoint"""
    print("\n📸 Testing /capture_person endpoint...")
    try:
        data = {"name": "Test User", "view": "front"}
        response = requests.post(
            f"{BASE_URL}/capture_person",
            json=data,
            timeout=10
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code in [200, 400]  # 400 is ok if no face detected
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_clear_person():
    """Test /clear_person endpoint"""
    print("\n🗑️  Testing /clear_person endpoint...")
    try:
        response = requests.post(f"{BASE_URL}/clear_person", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("="*60)
    print("🧪 Person Tracking Endpoints Test")
    print("="*60)
    print(f"\nTesting endpoints at: {BASE_URL}")
    print("\n⚠️  Make sure web_voice_interface.py is running first!")
    
    results = {
        "person_status": test_person_status(),
        "capture_person": test_capture_person(),
        "clear_person": test_clear_person(),
        "person_status_after_clear": test_person_status()
    }
    
    print("\n" + "="*60)
    print("📊 Test Results Summary")
    print("="*60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above.")
    print("="*60)

if __name__ == "__main__":
    main()
