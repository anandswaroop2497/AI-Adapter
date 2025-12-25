# AI Adapter - Robust & Scalable Computer Vision Platform

A flexible, production-ready AI inference adapter designed for multi-model, multi-task computer vision applications. Built with extensibility and maintainability at its core.

## 🎯 Project Overview

This project implements a **scalable AI adapter architecture** that decouples inference logic from application code, enabling easy integration of multiple AI models and tasks without tight coupling. The system currently supports YOLOv8-based person detection and counting, with a design that makes adding new models and tasks straightforward.

### Key Features

- ✅ **Multi-Model Support**: Plug-and-play architecture for multiple AI models
- ✅ **Multi-Task Per Model**: One model can serve multiple use cases
- ✅ **RESTful API**: FastAPI-based HTTP interface for easy integration
- ✅ **Dynamic Task Configuration**: Enable/disable tasks without code changes
- ✅ **ONNX Runtime**: Optimized inference with cross-platform compatibility
- ✅ **Active & Passive Modes**: Flexible camera handling with fallback support
- ✅ **Production-Ready**: NMS, confidence thresholding, and error handling built-in

---

## 🏗️ Architecture & Design Philosophy

### The Model Handler Pattern

The core architectural pattern that makes this system robust and scalable:

```
┌─────────────────────────────────────────────────────────┐
│                     FastAPI Server                       │
│                    (adapter/main.py)                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ Routes tasks via registry
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Model Registry (Dictionary)                 │
│   {                                                      │
│     "person_detection": YOLOv8Handler instance,         │
│     "person_counting": YOLOv8Handler instance,          │
│     "object_detection": FutureModelHandler instance     │
│   }                                                      │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐   ┌──────────────────┐
│  YOLOv8Handler   │   │ Future Handlers  │
│                  │   │ (ResNet, etc.)   │
├──────────────────┤   ├──────────────────┤
│ • Detection      │   │ • Classification │
│ • Counting       │   │ • Segmentation   │
│ • NMS            │   │ • ...            │
└──────────────────┘   └──────────────────┘
```

### Design Principles

1. **Separation of Concerns**
   - **Adapter Layer**: HTTP interface and routing (`adapter/main.py`)
   - **Model Handlers**: Inference logic (`adapter/models/`)
   - **Configuration**: Centralized settings (`adapter/config.py`)
   - **Utilities**: Reusable helpers (`adapter/utils/`)

2. **Abstract Base Class Pattern**
   - All handlers inherit from `BaseModelHandler`
   - Enforces consistent interface: `get_supported_tasks()` and `infer()`
   - Makes adding new models trivial (just implement the interface)

3. **Task-Based Routing**
   - Tasks are first-class citizens (e.g., "person_detection", "person_counting")
   - One model can handle multiple tasks
   - Dynamic task registration at startup

4. **Configuration Over Code**
   - Enable/disable tasks via `ENABLED_TASKS` dict
   - Model paths in `MODEL_CONFIGS`
   - No code changes needed for common adjustments

---

## 📁 Project Structure

```
adaplayer/
├── adapter/                    # Core AI adapter
│   ├── main.py                # FastAPI server & task routing
│   ├── config.py              # Configuration & model registry
│   ├── models/                # Model handler implementations
│   │   ├── __init__.py
│   │   ├── base_handler.py   # Abstract base class
│   │   └── yolov8_handler.py # YOLOv8 implementation
│   └── utils/                 # Helper utilities
│       └── image_utils.py    # Image loading & preprocessing
│
├── kavach/                    # Camera runner client
│   └── runner.py             # Capture frames & call adapter API
│
├── frames/                    # Runtime: captured frames stored here
│   └── camera_{id}/
│       └── latest.jpg
│
├── yolov8n.onnx              # YOLOv8 nano model (ONNX format)
├── requirements.txt          # Python dependencies
├── run_task.bat             # Windows helper script
└── README.md                # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Webcam (optional, supports passive mode without camera)
- Windows/Linux/macOS

### 1. Installation

```bash
# Clone the repository
cd adaplayer

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the Adapter Server

```bash
uvicorn adapter.main:app --reload --port 9100
```

Expected output:
```
🚀 Initializing AI Adapter...
  ✓ Registered task: person_detection
  ✓ Registered task: person_counting
✅ Adapter ready with 2 active task(s)

INFO: Uvicorn running on http://127.0.0.1:9100
```

### 3. Run the Camera Runner

In a new terminal:

```bash
# Activate venv first
venv\Scripts\activate

# Run person detection
python kavach/runner.py --task person_detection

# Or run person counting
python kavach/runner.py --task person_counting

# List available tasks
python kavach/runner.py --list-tasks
```

**Alternative: Use the batch file (Windows)**
```bash
run_task.bat --task person_detection
```

---

## 📡 API Reference

### Health Check
```http
GET /health
```
Returns: `{"status": "ok"}`

### Get Capabilities
```http
GET /capabilities
```
Returns:
```json
{
  "tasks": ["person_detection", "person_counting"]
}
```

### Run Inference
```http
POST /infer
Content-Type: application/json

{
  "task": "person_detection",
  "input": {
    "frame": {
      "uri": "kavach://frames/camera_0/latest.jpg"
    }
  }
}
```

**Response (Detection):**
```json
{
  "bbox": [100, 50, 200, 400],
  "confidence": 0.87,
  "message": "Person detected with confidence 0.87"
}
```

**Response (Counting):**
```json
{
  "count": 3,
  "confidence": 0.82,
  "detections": [
    {"bbox": [10, 20, 100, 200], "confidence": 0.85},
    {"bbox": [150, 30, 120, 220], "confidence": 0.81},
    {"bbox": [300, 40, 110, 210], "confidence": 0.79}
  ]
}
```

---

## 🔧 Configuration

### Adjust Confidence Threshold

Edit `adapter/config.py`:
```python
CONFIDENCE_THRESHOLD = 0.5  # Adjust between 0.0 - 1.0
```

### Enable/Disable Tasks

```python
ENABLED_TASKS = {
    "person_detection": True,   # Set to False to disable
    "person_counting": True,
}
```

### Change Input Size

```python
INPUT_SIZE = 640  # YOLOv8 default, can be 320/416/640
```

---

## 🧩 Adding New Models

The architecture makes adding new models straightforward:

### Step 1: Create a Handler

Create `adapter/models/new_model_handler.py`:

```python
from .base_handler import BaseModelHandler
from typing import List, Dict, Any

class NewModelHandler(BaseModelHandler):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        # Load your model here
        
    def get_supported_tasks(self) -> List[str]:
        return ["task1", "task2"]
    
    def infer(self, task: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if task == "task1":
            return self._do_task1(input_data)
        elif task == "task2":
            return self._do_task2(input_data)
```

### Step 2: Register in Config

Edit `adapter/config.py`:

```python
MODEL_CONFIGS = {
    "yolov8n": {...},
    "new_model": {
        "path": "path/to/model.onnx",
        "handler_class": "NewModelHandler",
    }
}

ENABLED_TASKS = {
    "person_detection": True,
    "person_counting": True,
    "task1": True,  # New tasks
    "task2": True,
}
```

### Step 3: Import and Initialize

Edit `adapter/models/__init__.py`:
```python
from .new_model_handler import NewModelHandler
__all__ = [..., "NewModelHandler"]
```

Edit `adapter/main.py` startup:
```python
new_model_handler = NewModelHandler(MODEL_CONFIGS["new_model"]["path"])
for task in new_model_handler.get_supported_tasks():
    if ENABLED_TASKS.get(task, False):
        model_registry[task] = new_model_handler
```

That's it! Your new model is now integrated. 🎉

---

## 🎥 Camera Runner Features

### Active Mode
Captures frames from camera and sends to adapter:
```bash
python kavach/runner.py --task person_detection --camera 0 --interval 2.0
```

### Passive Mode
Automatically activates if camera is busy. Monitors file changes:
```
⚠️  Camera 0 is busy or unavailable.
🔄 Switching to PASSIVE MODE: Watching frames/camera_0/latest.jpg...
```

### Multiple Tasks
Run multiple tasks on the same frame:
```bash
python kavach/runner.py --task person_detection,person_counting
```

---

## 🧪 Testing

### Verify Setup
```bash
python verify_setup.py
```

### Check Server Health
```bash
curl http://127.0.0.1:9100/health
```

### Test Inference
```bash
curl -X POST http://127.0.0.1:9100/infer \
  -H "Content-Type: application/json" \
  -d '{"task":"person_detection","input":{"frame":{"uri":"kavach://frames/camera_0/latest.jpg"}}}'
```

---

## 🛠️ Technical Details

### Technologies Used

- **FastAPI**: Modern, high-performance web framework
- **ONNX Runtime**: Cross-platform, optimized inference engine
- **OpenCV**: Image processing and camera capture
- **YOLOv8**: State-of-the-art object detection
- **httpx**: Async HTTP client for runner
- **NumPy**: Numerical computing

### Key Implementations

1. **Non-Maximum Suppression (NMS)**
   - Removes duplicate detections
   - IoU threshold: 0.45
   - See: `YOLOv8Handler._apply_nms()`

2. **Dynamic Task Registration**
   - Tasks registered at startup based on config
   - Enables hot-swapping via server restart

3. **URI-Based Image Loading**
   - Custom URI scheme: `kavach://frames/...`
   - Decouples storage from inference logic

4. **Error Handling**
   - HTTPException for user-facing errors
   - Graceful degradation (passive mode)
   - Timeout handling in runner

---

## 📊 Performance

- **Inference Time**: ~50-150ms on CPU (YOLOv8n)
- **Detection Accuracy**: 0.5+ confidence threshold
- **NMS IoU**: 0.45 threshold
- **Memory**: ~200MB with model loaded

---

## 🔮 Future Enhancements

- [ ] GPU acceleration (CUDA/TensorRT)
- [ ] Multiple camera support
- [ ] WebSocket streaming
- [ ] Model versioning
- [ ] Metrics & monitoring
- [ ] Docker containerization
- [ ] Face recognition models
- [ ] Object tracking across frames

---

## 🤝 Contributing

This architecture is designed for extensibility. To contribute:

1. Follow the Model Handler Pattern
2. Inherit from `BaseModelHandler`
3. Update configuration files
4. Add tests for new functionality
5. Document new tasks in this README

---

## 📝 License

This project is open-source. Feel free to use and modify.

---

## 👨‍💻 Author

Built with a focus on **robust, scalable, and maintainable** code architecture.

### Design Goals Achieved:

✅ **Robust**: Abstract interfaces, error handling, NMS, confidence thresholds  
✅ **Scalable**: Add models without touching existing code  
✅ **Maintainable**: Clear separation of concerns, configuration over code  
✅ **Production-Ready**: FastAPI, ONNX Runtime, proper async handling  

---

## 📞 Support

For questions or issues, check:
- Configuration: `adapter/config.py`
- Add models: See "Adding New Models" section
- API errors: Check `/health` endpoint
- Runner issues: Try `--list-tasks` flag

---

**Happy Coding! 🚀**
