import cv2
import time
import httpx
import os


cap = cv2.VideoCapture(0)


FRAME_DIR = r"D:\Ageis AI\Ai Adapter\frames\camera_1"
os.makedirs(FRAME_DIR, exist_ok=True)

print(f"Camera Runner Started. Saving to: {FRAME_DIR}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not accessible")
        break

    
    frame_path = os.path.join(FRAME_DIR, "latest.jpg")
    cv2.imwrite(frame_path, frame)  
    
    payload = {
        "task": "person_detection",
        "model_id": "person-detector-v1",
        "camera_id": "camera_1",
        "timestamp_ms": int(time.time() * 1000),
        "input": {
            "type": "video_frame",
            "frame": {
                
                "uri": "kavach://frames/camera_1/latest.jpg"
            }
        },
        "privacy": {"data_egress_class": "CLASS_0"}
    }

   
    try:
        r = httpx.post(
            "http://127.0.0.1:9100/v1/infer",
            json=payload,
            timeout=5
        )
        print(f"Response: {r.json()}")
    except Exception as e:
        print(f"Adapter Error: {e}")

    time.sleep(2) 