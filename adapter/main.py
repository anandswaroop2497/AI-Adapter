from fastapi import FastAPI, HTTPException
from typing import Dict
import time

from .models import YOLOv8Handler
from .config import MODEL_CONFIGS, ENABLED_TASKS

app = FastAPI()

# Model registry - maps task names to handler instances
model_registry: Dict[str, any] = {}


@app.on_event("startup")
async def startup_event():
    """
    Initialize model handlers and build task registry on startup.
    Only registers tasks that are enabled in config.
    """
    print("\n🚀 Initializing AI Adapter...")
    
    # Initialize YOLOv8 handler
    yolov8_config = MODEL_CONFIGS["yolov8n"]
    yolov8_handler = YOLOv8Handler(yolov8_config["path"])
    
    # Register enabled tasks
    for task in yolov8_handler.get_supported_tasks():
        if ENABLED_TASKS.get(task, False):  # Only register if enabled
            model_registry[task] = yolov8_handler
            print(f"  ✓ Registered task: {task}")
        else:
            print(f"  ✗ Skipped task (disabled): {task}")
    
    print(f"\n✅ Adapter ready with {len(model_registry)} active task(s)\n")


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/capabilities")
def capabilities():
    """
    Return list of supported tasks.
    Dynamically generated based on registered handlers and enabled tasks.
    """
    return {"tasks": list(model_registry.keys())}


@app.post("/infer")
def infer(req: dict):
    """
    Main inference endpoint.
    Routes requests to appropriate model handler based on task name.
    
    Request format:
    {
        "task": "person_detection" or "person_counting",
        "input": {
            "frame": {
                "uri": "kavach://frames/camera1/frame.jpg"
            }
        }
    }
    """
    # Validate request
    task = req.get("task")
    if not task:
        raise HTTPException(status_code=400, detail="Missing 'task' field")
    
    # Check if task is supported and enabled
    if task not in model_registry:
        available_tasks = list(model_registry.keys())
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported or disabled task: '{task}'. Available tasks: {available_tasks}"
        )
    
    # Validate input
    if "input" not in req or "frame" not in req["input"]:
        raise HTTPException(status_code=400, detail="Missing 'input.frame' field")
    
    # Get handler and run inference
    handler = model_registry[task]
    
    try:
        result = handler.infer(task, req["input"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")