# AI_ADAPTER

<<<<<<< HEAD
An adapter layer for VoIP cameras that enables "bring your own model" functionality. Camera applications capture images and apply user-selected AI models through a unified API interface.
=======
An adapter layer for VoIP cameras that enables "bring your own model" functionality.
>>>>>>> 404a5612faf9e2720e36cd88a30e10ffb5255b7f

## What is this?

This is an adapter layer that sits between VoIP cameras and AI models. It provides:
- **RESTful API endpoints** for inference requests
- **Model management** through a handler pattern
- **Scalable architecture** for adding new models easily
<<<<<<< HEAD

The camera app captures frames and sends them to this adapter, which runs the selected AI model and returns results.
=======
>>>>>>> 404a5612faf9e2720e36cd88a30e10ffb5255b7f

---

## Main Components

### 1. **Adapter Server** (`adapter/`)
The core FastAPI server that:
- Exposes `/infer`, `/capabilities`, and `/health` endpoints
- Routes inference requests to the appropriate model handler
- Manages model registry and task assignments

**Key Files:**
- `main.py` - FastAPI server with API routes
- `config.py` - Model configurations and task registry
- `models/` - Model handler implementations

### 2. **Model Handlers** (`adapter/models/`)
Each handler implements inference logic for a specific model type:
- `base_handler.py` - Abstract base class defining the interface
- `yolov8_handler.py` - YOLOv8 implementation (person detection/counting)

All handlers inherit from `BaseModelHandler` and implement:
```python
get_supported_tasks() -> List[str]  # What tasks can this model do?
infer(task, input_data) -> Dict     # Run inference
```

### 3. **Camera Runner** (`kavach/runner.py`)
Client application that:
- Captures frames from camera (or monitors existing frames)
- Sends inference requests to the adapter API
- Displays results

### 4. **Utilities** (`adapter/utils/`)
Helper functions for:
- Image loading and preprocessing
- URI parsing (`kavach://frames/...`)

---

## How It Works

```
┌─────────────┐                  ┌──────────────┐                 ┌─────────────┐
│   Camera    │  Capture Image   │   Adapter    │  Model Handler  │    Model    │
│     App     │ ───────────────> │   API        │ ─────────────>  │   (ONNX)    │
│             │  POST /infer     │              │                 │             │
│             │ <─────────────── │              │ <───────────────│             │
└─────────────┘  Return Result   └──────────────┘  Inference      └─────────────┘
```

**Flow:**
1. Camera captures image → saves to `frames/camera_X/latest.jpg`
2. Runner sends POST request to `/infer` with task name and image URI
3. Adapter routes to appropriate model handler
4. Handler runs model inference and returns results
5. Camera app receives results (bounding boxes, counts, etc.)

---

## Architecture for Scalability

### Model Handler Pattern
The architecture uses a **registry pattern** to decouple models from routing logic:

```python
# In config.py
model_registry = {
    "person_detection": yolov8_handler_instance,
    "person_counting": yolov8_handler_instance,
    "future_task": future_model_handler_instance
}
```

**Why this scales:**
- ✅ Add new models without modifying API routes
- ✅ One model can support multiple tasks
- ✅ Enable/disable tasks via configuration
- ✅ Models are loaded once and reused

### Adding a New Model

**Step 1:** Create handler in `adapter/models/your_model_handler.py`
```python
from .base_handler import BaseModelHandler

class YourModelHandler(BaseModelHandler):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        # Load your model
        
    def get_supported_tasks(self):
        return ["task_name_1", "task_name_2"]
    
    def infer(self, task, input_data):
        # Implement inference logic
        return {"result": "..."}
```

**Step 2:** Register in `adapter/config.py`
```python
MODEL_CONFIGS = {
    "your_model": {
        "path": "path/to/model.onnx",
        "handler_class": "YourModelHandler"
    }
}

ENABLED_TASKS = {
    "task_name_1": True,
    "task_name_2": True
}
```

**Step 3:** Initialize in `adapter/main.py`
```python
from adapter.models import YourModelHandler

your_handler = YourModelHandler(MODEL_CONFIGS["your_model"]["path"])
for task in your_handler.get_supported_tasks():
    if ENABLED_TASKS.get(task):
        model_registry[task] = your_handler
```

Done! Your model is now available through the API.

---

## Project Structure

```
AI_ADAPTER/
├── adapter/                    # Core adapter layer
│   ├── main.py                 # FastAPI server + routes
│   ├── config.py               # Model registry & configuration
│   ├── models/                 # Model handler implementations
│   │   ├── base_handler.py     # Abstract base class
│   │   └── yolov8_handler.py   # YOLOv8 implementation
│   └── utils/
│       └── image_utils.py      # Image preprocessing
│
├── kavach/                     # Camera client
│   └── runner.py               # Frame capture + API calls
│
├── frames/                     # Runtime: captured images
│   └── camera_X/latest.jpg
│
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## Quick Start

### 1. Install Dependencies
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Start Adapter Server
```bash
uvicorn adapter.main:app --reload --port 9100
```

### 3. Run Camera Client
```bash
python kavach/runner.py --task person_detection

# List available tasks
python kavach/runner.py --list-tasks
```

---

## API Endpoints

### `GET /health`
Check if server is running
```json
{"status": "ok"}
```

### `GET /capabilities`
List available tasks
```json
{
  "tasks": ["person_detection", "person_counting"]
}
```

### `POST /infer`
Run inference on an image
```json
{
  "task": "person_detection",
  "input": {
    "frame": {
      "uri": "kavach://frames/camera_0/latest.jpg"
    }
  }
}
```

**Response:**
```json
{
  "bbox": [x, y, width, height],
  "confidence": 0.87,
  "message": "Person detected with confidence 0.87"
}
```

---

## Configuration

Edit `adapter/config.py` to:
- Adjust confidence thresholds
- Enable/disable tasks
- Add new models
- Configure input sizes

---

## Current Models

### YOLOv8 Handler
**Tasks:**
- `person_detection` - Detect single person with highest confidence
- `person_counting` - Count all persons in frame

**Features:**
- Non-Maximum Suppression (NMS) for duplicate removal
- Configurable confidence threshold
- ONNX Runtime for optimized inference

---

## Technologies

- **FastAPI** - API server framework
- **ONNX Runtime** - Model inference engine
- **OpenCV** - Image processing & camera capture
- **NumPy** - Array operations
- **httpx** - HTTP client for runner

---

**Built for scalability and extensibility.**
