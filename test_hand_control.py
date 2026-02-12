#!/usr/bin/env python3
"""
Test script for hand and arm control
Tests serial communication with Arduino
"""

import sys
import time

try:
    from serial_bridge import RobotBridge
    from config import SERIAL_PORT, SERIAL_BAUD
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project directory")
    sys.exit(1)

def test_hands(bot):
    """Test hand controls"""
    print("\n" + "="*50)
    print("🖐️  TESTING HAND CONTROLS")
    print("="*50)
    
    tests = [
        ("Open left hand", bot.open_left_hand),
        ("Close left hand", bot.close_left_hand),
        ("Open right hand", bot.open_right_hand),
        ("Close right hand", bot.close_right_hand),
    ]
    
    for name, func in tests:
        print(f"\n{name}...")
        func()
        time.sleep(1)
    
    print("\n✅ Hand controls test complete")

def test_wrists(bot):
    """Test wrist controls"""
    print("\n" + "="*50)
    print("🔄 TESTING WRIST CONTROLS")
    print("="*50)
    
    angles = [0, 45, 90, 135, 180, 90]
    
    print("\nTesting left wrist...")
    for angle in angles:
        print(f"  Setting to {angle}°")
        bot.set_left_wrist(angle)
        time.sleep(0.5)
    
    print("\nTesting right wrist...")
    for angle in angles:
        print(f"  Setting to {angle}°")
        bot.set_right_wrist(angle)
        time.sleep(0.5)
    
    print("\n✅ Wrist controls test complete")

def test_biceps(bot):
    """Test bicep controls"""
    print("\n" + "="*50)
    print("💪 TESTING BICEP CONTROLS")
    print("="*50)
    
    print("\nLeft bicep up...")
    bot.left_bicep_up()
    time.sleep(3)
    
    print("Left bicep down...")
    bot.left_bicep_down()
    time.sleep(3)
    
    print("Right bicep up...")
    bot.right_bicep_up()
    time.sleep(3)
    
    print("Right bicep down...")
    bot.right_bicep_down()
    time.sleep(3)
    
    print("Stopping all biceps...")
    bot.stop_biceps()
    
    print("\n✅ Bicep controls test complete")

def main():
    print("="*50)
    print("🤖 HAND & ARM CONTROL TEST")
    print("="*50)
    print(f"\nConnecting to: {SERIAL_PORT} @ {SERIAL_BAUD} baud")
    
    try:
        bot = RobotBridge(port=SERIAL_PORT, baud_rate=SERIAL_BAUD)
        
        if not bot.ser:
            print("❌ Failed to connect to robot")
            print("Make sure:")
            print("  1. Arduino is connected")
            print("  2. Correct serial port in config.py")
            print("  3. Arduino is running the hand control sketch")
            return
        
        print("\n✅ Connected successfully!")
        print("\nSelect test:")
        print("1. Test hands only")
        print("2. Test wrists only")
        print("3. Test biceps only")
        print("4. Test all (full sequence)")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            test_hands(bot)
        elif choice == '2':
            test_wrists(bot)
        elif choice == '3':
            test_biceps(bot)
        elif choice == '4':
            test_hands(bot)
            test_wrists(bot)
            test_biceps(bot)
        elif choice == '5':
            print("Exiting...")
        else:
            print("Invalid choice")
        
        bot.close()
        print("\n" + "="*50)
        print("✅ Test complete - connection closed")
        print("="*50)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        if 'bot' in locals():
            bot.close()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
