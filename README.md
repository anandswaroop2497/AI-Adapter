## Project Overview
This project implements the first working AI integration for the Aegis Kavach ecosystem. It establishes a pipeline where RTSP/Camera frames are processed via an ONNX AI Adapter to perform Person Detection on the Edge (CPU).
## Project Structure
```text
AI_ADAPTER/
├── adapter/
│   ├── main.py            # FastAPI Service (The AI Brain)
│   ├── yolov8n.onnx       # The AI Model (Must be exported/downloaded)
│   └── __pycache__/
├── frames/                # Storage for active frames
│   └── camera_1/
│       └── latest.jpg     # The current frame being analyzed
├── kavach/
│   └── runner.py          # The Camera Agent (Captures & Requests)
├── venv/                  # Python Virtual Environment
└── requirements.txt       # Dependencies
```

## How To Run
Step 1:Start the Ai Adapter
```
cd adapter
uvicorn main:app --port 9100 --reload
```

Step 2:Start the runner
```
cd kavach
python runner.py
```
## API Endpoints
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `health` | Service health check |
| `GET` | `capabilities` | Lists supported tasks (`["person_detection"]`) |
| `POST` | `infer` | Main inference endpoint. Accepts KAI-C formatted JSON. |
