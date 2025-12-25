# Quick Reference: Task Configuration

## Enabling/Disabling Tasks

You mentioned you want to run tasks based on your preference. Here's how:

### Option 1: Using Config File (RECOMMENDED)

Edit `adapter/config.py`:

```python
# Run BOTH tasks
ENABLED_TASKS = {
    "person_detection": True,
    "person_counting": True,
}

# Run ONLY detection
ENABLED_TASKS = {
    "person_detection": True,
    "person_counting": False,
}

# Run ONLY counting
ENABLED_TASKS = {
    "person_detection": False,
    "person_counting": True,
}
```

After changing, restart the server:
```bash
uvicorn adapter.main:app --reload --port 8000
```

The server will automatically register only the enabled tasks!

---

## API Usage

### Check Available Tasks
```bash
GET http://localhost:8000/capabilities
```

Returns only the enabled tasks:
```json
{"tasks": ["person_detection", "person_counting"]}
```

### Run Person Detection
```bash
POST http://localhost:8000/infer
{
  "task": "person_detection",
  "input": {
    "frame": {"uri": "kavach://frames/camera_1/latest.jpg"}
  }
}
```

### Run Person Counting
```bash
POST http://localhost:8000/infer
{
  "task": "person_counting",
  "input": {
    "frame": {"uri": "kavach://frames/camera_1/latest.jpg"}
  }
}
```

---

## Adding New Models in Future

1. Create handler in `adapter/models/your_model_handler.py`
2. Add config in `adapter/config.py`
3. Register in `adapter/main.py` startup
4. Done!

See [walkthrough.md](file:///C:/Users/hp/.gemini/antigravity/brain/7e3f5abc-9298-4863-a8fe-896c7616c873/walkthrough.md) for detailed examples.
