# 🎥 Quick Start: Running Tasks

The runner has been updated! You can now easily switch between tasks.

## Simple Commands

### List what tasks are available:
```bash
run_task.bat --list-tasks
```

### Run Person Detection:
```bash
run_task.bat --task person_detection
```

### Run Person Counting:
```bash
run_task.bat --task person_counting
```

### Stop the runner:
Press `Ctrl+C`

## What Happens

The runner will:
1. ✅ Open your camera (camera 0 by default)
2. ✅ Capture frames every 2 seconds
3. ✅ Send them to the adapter with your selected task
4. ✅ Print results in real-time

## Switch Tasks Anytime

**Need detection now?**
```bash
run_task.bat --task person_detection
```

**Want to count persons instead?** Just stop (Ctrl+C) and run:
```bash
run_task.bat --task person_counting
```

It's that easy! No code changes needed.

## Example Output

### Person Detection
```
[Frame 0001] ✅ Detection | Conf: 0.82 | BBox: [0, 213, 598, 266] | Latency: 289ms
[Frame 0002] ✅ Detection | Conf: 0.79 | BBox: [5, 210, 595, 270] | Latency: 295ms
```

### Person Counting
```
[Frame 0001] ✅ Counting  | Count: 2 | Avg Conf: 0.81 | Latency: 305ms
               Detections: [{'bbox': [0, 213, 598, 266], 'confidence': 0.82}, ...]
```

## More Options

See [RUNNER_GUIDE.md](file:///c:/Users/hp/Desktop/adaplayer/RUNNER_GUIDE.md) for advanced options like:
- Using a different camera
- Changing capture interval
- And more!
