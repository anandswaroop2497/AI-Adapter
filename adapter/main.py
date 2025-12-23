from fastapi import FastAPI, HTTPException
import onnxruntime as ort
import cv2
import numpy as np
import time
import os

app = FastAPI()


model_filename = "yolov8n.onnx"


session = ort.InferenceSession(model_filename, providers=['CPUExecutionProvider'])
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/capabilities")
def capabilities():
    return {"tasks": ["person_detection"]}

@app.post("/infer")
def infer(req: dict):
    
    if req.get("task") != "person_detection":
        raise HTTPException(400, "Unsupported task")

    uri = req["input"]["frame"]["uri"]
    BASE_FRAMES_DIR = r"D:\Ageis AI\Ai Adapter\frames"
    
   
    relative_path = uri.replace("kavach://frames/", "").replace("/", os.sep)
    frame_path = os.path.join(BASE_FRAMES_DIR, relative_path)

    if not os.path.exists(frame_path):
        raise HTTPException(404, f"Frame not found: {frame_path}")

    
    img = cv2.imread(frame_path)
    if img is None: raise HTTPException(400, "Invalid image")

    h_img, w_img = img.shape[:2]

    
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (640, 640), swapRB=True, crop=False)

    start = time.time()
    
    
    outputs = session.run(None, {session.get_inputs()[0].name: blob})
    
    
    predictions = np.transpose(outputs[0], (0, 2, 1)).squeeze()
    
    
    person_rows = predictions[predictions[:, 4] > 0.5]

    result = {
        "label": "person",
        "confidence": 0.0,
        "bbox": [0, 0, 0, 0],
        "executed_at": int(time.time() * 1000),
        "latency_ms": 0
    }

    if len(person_rows) > 0:
        
        best_row = person_rows[np.argmax(person_rows[:, 4])]
        cx, cy, w, h = best_row[:4]
        conf = float(best_row[4])


        
        if w < 1.0: 
            # Normalized case
            left   = int((cx - w/2) * w_img)
            top    = int((cy - h/2) * h_img)
            width  = int(w * w_img)
            height = int(h * h_img)
        else:
            
            x_scale = w_img / 640
            y_scale = h_img / 640
            left   = int((cx - w/2) * x_scale)
            top    = int((cy - h/2) * y_scale)
            width  = int(w * x_scale)
            height = int(h * y_scale)

        
        result["bbox"] = [max(0, left), max(0, top), width, height]
        result["confidence"] = round(conf, 2)

    result["latency_ms"] = int((time.time() - start) * 1000)
    
    
    print(f"Debug: Conf={result['confidence']}, BBox={result['bbox']}, Raw_W={w}")
    
    return result