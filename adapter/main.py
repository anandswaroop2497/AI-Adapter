"""
FastAPI Application - Main Entry Point for AI Adapter API

This file defines the REST API server that:
1. Loads model handlers at startup
2. Registers tasks to handlers in a registry (Dictionary mapping)
3. Provides endpoints for health checks, capabilities, and inference
4. Routes incoming requests to the appropriate handler based on task name

KEY CONCEPTS:
- model_registry: Dictionary that maps task names → handler instances
- Startup event: Loads models when server starts (not on every request)  
- Routing: Uses task name from request to find the right handler

ENDPOINTS:
- GET  /health        - Check if server is running
- GET  /capabilities  - List all available tasks
- POST /infer         - Run model inference for a specific task
"""

from fastapi import FastAPI, HTTPException
from typing import Dict
import time

# Import model handlers and configuration
from .models import YOLOv8Handler
from .config import MODEL_CONFIGS, ENABLED_TASKS

# Create FastAPI application instance
# This is the main app object that handles all HTTP requests
app = FastAPI()

# MODEL REGISTRY - The Core Routing Mechanism
# This dictionary maps task names (strings) to handler instances (objects)
# Example after startup:
# {
#     "person_detection": <YOLOv8Handler instance>,
#     "person_counting": <YOLOv8Handler instance>  # Same instance!
# }
# When a request comes in with task="person_counting", we do:
# handler = model_registry["person_counting"]  # Get handler
# result = handler.infer("person_counting", input_data)  # Run it
model_registry: Dict[str, any] = {}


@app.on_event("startup")
async def startup_event():
    """
    Server startup handler - Runs ONCE when server starts.
    
    This function:
    1. Creates model handler instances (loads models into memory)
    2. Asks each handler what tasks it supports
    3. Registers enabled tasks in the model_registry
    
    WHY AT STARTUP?
    - Models are loaded ONCE, not on every request (much faster!)
    - Registry is built before any requests arrive
    - Validates models load correctly before accepting traffic
    
    FLOW:
    1. Create YOLOv8Handler(model_path) → Loads ONNX model
    2. handler.get_supported_tasks() → ["person_detection", "person_counting"]
    3. For each task, if ENABLED_TASKS[task] == True:
       - model_registry[task] = handler
    4. Server is ready to accept requests
    """
    print("\n🚀 Initializing AI Adapter...")
    
    # Step 1: Load YOLOv8 model configuration from config.py
    yolov8_config = MODEL_CONFIGS["yolov8n"]  # Get model path and settings
    
    # Step 2: Create handler instance - this LOADS the model file
    # The model is loaded into memory here (expensive operation)
    yolov8_handler = YOLOv8Handler(yolov8_config["path"])
    
    # Step 3: Register tasks that this handler supports
    # get_supported_tasks() returns ["person_detection", "person_counting"]
    for task in yolov8_handler.get_supported_tasks():
        # Check if this task is enabled in config.py
        if ENABLED_TASKS.get(task, False):  # Only register if enabled
            # Add to registry: task name → handler instance
            model_registry[task] = yolov8_handler
            print(f"  ✓ Registered task: {task}")
        else:
            # Task is supported by model but disabled in config
            print(f"  ✗ Skipped task (disabled): {task}")
    
    # Startup complete!
    print(f"\n✅ Adapter ready with {len(model_registry)} active task(s)\n")


@app.get("/health")
def health():
    """
    Health check endpoint.
    
    Simple endpoint to verify the server is running and responding.
    Used by monitoring systems, load balancers, etc.
    
    Returns:
        {"status": "ok"} if server is alive
    
    USAGE:
        curl http://localhost:9100/health
    """
    return {"status": "ok"}


@app.get("/capabilities")
def capabilities():
    """
    Return list of supported tasks.
    
    Dynamically generates list based on:
    - Which handlers are loaded
    - Which tasks are enabled in config
    
    This is determined at runtime from model_registry.keys(),
    so it always reflects the current state.
    
    Returns:
        {"tasks": ["person_detection", "person_counting", ...]}
    
    USAGE:
        curl http://localhost:9100/capabilities
    """
    return {"tasks": list(model_registry.keys())}


@app.post("/infer")
def infer(req: dict):
    """
    Main inference endpoint - Routes requests to appropriate handler.
    
    This is the MAIN API endpoint for running model inference.
    It works as a ROUTER that:
    1. Extracts task name from request
    2. Looks up handler in registry
    3. Calls handler.infer() with the task and input
    4. Returns result
    
    REQUEST FORMAT:
    {
        "task": "person_detection" or "person_counting",
        "input": {
            "frame": {
                "uri": "kavach://frames/camera_0/latest.jpg"
            }
        }
    }
    
    RESPONSE FORMAT (depends on task):
    {
        "task": "person_counting",
        "count": 2,
        "confidence": 0.75,
        "detections": [...],
        "annotated_image_uri": "kavach://frames/camera_0/annotated.jpg",
        "executed_at": 1234567890,
        "latency_ms": 150
    }
    
    FLOW:
    Client → POST /infer → Extract task name → model_registry[task] → handler.infer() → Result
    
    Args:
        req: Request dictionary containing task and input fields
        
    Returns:
        Result dictionary from handler.infer()
        
    Raises:
        HTTPException 400: Invalid request (missing fields, unsupported task)
        HTTPException 500: Inference execution error
    """
    # Step 1: Validate request has 'task' field
    task = req.get("task")
    if not task:
        raise HTTPException(status_code=400, detail="Missing 'task' field")
    
    # Step 2: Check if task is supported and enabled
    # This checks if task exists as a key in model_registry
    if task not in model_registry:
        available_tasks = list(model_registry.keys())
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported or disabled task: '{task}'. Available tasks: {available_tasks}"
        )
    
    # Step 3: Validate request has 'input.frame' field
    if "input" not in req or "frame" not in req["input"]:
        raise HTTPException(status_code=400, detail="Missing 'input.frame' field")
    
    # Step 4: Get the handler for this task from registry
    # Example: task="person_counting" → handler = <YOLOv8Handler instance>
    handler = model_registry[task]
    
    # Step 5: Run inference using the handler
    try:
        # Call handler.infer(task, input_data)
        # The handler will route internally to the right method:
        # - "person_detection" → _detect_persons()
        # - "person_counting" → _count_persons()
        result = handler.infer(task, req["input"])
        return result
    except HTTPException:
        # Re-raise HTTP exceptions as-is (from handler validation)
        raise
    except Exception as e:
        # Catch any other errors and return 500
        print(f"❌ Error during inference: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")