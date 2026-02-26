import cv2
import requests
import numpy as np
import os

from config import WEB_PORT, RASPBERRY_PI_IP

# Configuration — auto-detect HTTPS vs HTTP
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(SCRIPT_DIR, 'cert.pem')):
    STREAM_URL = f"https://127.0.0.1:{WEB_PORT}/video_feed"
    SSL_VERIFY = False  # Self-signed cert
else:
    STREAM_URL = f"http://127.0.0.1:{WEB_PORT}/video_feed"
    SSL_VERIFY = True
WINDOW_NAME = "NAVIS VISION SYSTEM"

def main():
    print(f"📡 Connecting to {STREAM_URL}...")
    
    # Suppress SSL warnings for self-signed certs
    if not SSL_VERIFY:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Create borderless fullscreen window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    import time
    
    while True:
        try:
            # Open the MJPEG stream
            stream = requests.get(STREAM_URL, stream=True, timeout=5, verify=SSL_VERIFY)
            if stream.status_code != 200:
                print(f"❌ Failed to reach camera stream. Retrying in 2 seconds...")
                time.sleep(2)
                continue
                
            bytes_data = b''
            print(f"✅ Connection Established! Press 'q' to quit.")
            
            for chunk in stream.iter_content(chunk_size=1024):
                bytes_data += chunk
                # Detect JPEG start/end flags
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')
                
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    
                    # Decode and display
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        # Enlarge it slightly to fill the screen better
                        frame = cv2.resize(frame, (800, 600))
                        cv2.imshow(WINDOW_NAME, frame)
                        
                    # Break on 'q'
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt
        except KeyboardInterrupt:
            print("⏹️ Manual quit requested.")
            break
        except Exception as e:
            print(f"⚠️ Stream Error: {e}. Retrying in 2 seconds...")
            time.sleep(2)
            
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
