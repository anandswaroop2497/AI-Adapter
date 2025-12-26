"""
Configuration for AI Adapter models and tasks.
"""
import os

# Base directory for the adapter
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_FRAMES_DIR = os.path.join(BASE_DIR, "..", "frames")

# Model configurations
MODEL_CONFIGS = {
    "yolov8n": {
        "path": os.path.join(BASE_DIR, "yolov8n.onnx"),
        "handler_class": "YOLOv8Handler",
    }
    # Add more models here as needed
    # "resnet": {
    #     "path": "resnet.onnx",
    #     "handler_class": "ResNetHandler",
    # }
}

# Task enablement - control which tasks are active
# Set to False to disable a task without removing code
ENABLED_TASKS = {
    "person_detection": True,
    "person_counting": True,
}

# Model inference settings
CONFIDENCE_THRESHOLD = 0.20  # Balanced - detects distant people, fewer false positives
INPUT_SIZE = 640
