# Kavach Runner - Quick Start Guide

The runner captures camera frames and runs your selected AI task on them.

## Basic Usage

### List Available Tasks
```bash
python kavach/runner.py --list-tasks
```

### Run Person Detection
```bash
python kavach/runner.py --task person_detection
```

### Run Person Counting
```bash
python kavach/runner.py --task person_counting
```

## Advanced Options

### Use Different Camera
```bash
python kavach/runner.py --task person_detection --camera 1
```

### Change Capture Interval (seconds)
```bash
python kavach/runner.py --task person_counting --interval 1.0
```

### Combine Options
```bash
python kavach/runner.py --task person_counting --camera 1 --interval 0.5
```

## Quick Switching Between Tasks

**Want detection now?**
```bash
python kavach/runner.py --task person_detection
```

**Switch to counting?** Just stop (Ctrl+C) and run:
```bash
python kavach/runner.py --task person_counting
```

## Output Examples

### Person Detection Output
```
[Frame 0001] ✅ Detection | Conf: 0.82 | BBox: [0, 213, 598, 266] | Latency: 289ms
[Frame 0002] ✅ Detection | Conf: 0.79 | BBox: [5, 210, 595, 270] | Latency: 295ms
```

### Person Counting Output
```
[Frame 0001] ✅ Counting  | Count: 2 | Avg Conf: 0.81 | Latency: 305ms
               Detections: [{'bbox': [0, 213, 598, 266], 'confidence': 0.82}, ...]
```

## Prerequisites

Make sure the adapter is running first:
```bash
uvicorn adapter.main:app --reload --port 8000
```

Then run the runner in a separate terminal.

## Stop Runner

Press `Ctrl+C` to stop gracefully.
