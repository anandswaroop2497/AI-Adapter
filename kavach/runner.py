"""
Kavach Camera Runner with configurable task selection.

Usage:
    python runner.py --task person_detection
    python runner.py --task person_counting
    python runner.py --help
"""
import cv2
import time
import httpx
import os
import argparse
import sys


def get_available_tasks():
    """Query the adapter for available tasks."""
    try:
        r = httpx.get("http://127.0.0.1:9100/capabilities", timeout=5)
        return r.json()["tasks"]
    except Exception as e:
        print(f"⚠️  Could not fetch capabilities from adapter: {e}")
        print("⚠️  Make sure the adapter is running: uvicorn adapter.main:app --reload --port 9100")
        return []


def run_camera(task: str, camera_id: int = 0, interval: float = 2.0):
    """
    Run camera capture and inference loop.
    
    Args:
        task: Task name(s) to run (e.g., "person_detection" or "person_detection,person_counting")
        camera_id: Camera device ID (default 0)
        interval: Seconds between captures (default 2.0)
    """
    # Parse tasks (support multiple comma-separated tasks)
    tasks = [t.strip() for t in task.split(',')]
    # Setup
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FRAME_DIR = os.path.join(BASE_DIR, "..", "frames", f"camera_{camera_id}")
    os.makedirs(FRAME_DIR, exist_ok=True)
    FRAME_PATH = os.path.join(FRAME_DIR, "latest.jpg")

    # Try to open camera
    cap = cv2.VideoCapture(camera_id)
    ret, _ = cap.read()
    
    is_passive = False
    if not ret:
        print(f"\n⚠️  Camera {camera_id} is busy or unavailable.")
        print(f"🔄 Switching to PASSIVE MODE: Watching {FRAME_PATH} for updates from another runner...")
        is_passive = True
        cap.release()
    else:
        print(f"\n🎥 ACTIVE MODE: Capturing from Camera {camera_id}")
    
    print(f"{'='*60}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Camera ID: {camera_id}")
    print(f"Frame Directory: {FRAME_DIR}")
    print(f"Interval: {interval}s")
    print(f"Adapter URL: http://127.0.0.1:9100/infer")
    print(f"{'='*60}\n")
    
    frame_count = 0
    last_file_mtime = 0
    
    try:
        while True:
            if not is_passive:
                # ACTIVE MODE: Capture and Save
                ret, frame = cap.read()
                if not ret:
                    print("❌ Camera lost. Stopping.")
                    break
                
                # Save frame
                cv2.imwrite(FRAME_PATH, frame)
            
            else:
                # PASSIVE MODE: Watch File
                if not os.path.exists(FRAME_PATH):
                    print(f"Waiting for {FRAME_PATH}...", end='\r')
                    time.sleep(1)
                    continue
                
                # Check if file updated
                try:
                    current_mtime = os.path.getmtime(FRAME_PATH)
                    if current_mtime <= last_file_mtime:
                        # File not changed yet, wait and retry
                        time.sleep(0.1)
                        continue
                    
                    last_file_mtime = current_mtime
                    # Allow writer to finish writing
                    time.sleep(0.1) 
                except OSError:
                    continue

            frame_count += 1
            
            # Run all tasks on this frame
            print(f"\n[Frame {frame_count:04d}] [{'PASSIVE' if is_passive else 'ACTIVE'}] {'='*40}")
            
            for task_name in tasks:
                # Prepare payload
                payload = {
                    "task": task_name,
                    "input": {
                        "frame": {
                            "uri": f"kavach://frames/camera_{camera_id}/latest.jpg"
                        }
                    }
                }
                
                # Call adapter
                try:
                    start_time = time.time()
                    r = httpx.post(
                        "http://127.0.0.1:9100/infer",
                        json=payload,
                        timeout=5
                    )
                    elapsed = int((time.time() - start_time) * 1000)
                    
                    if r.status_code == 200:
                        result = r.json()
                        
                        # Format output based on task
                        if task_name == "person_detection":
                            conf = result.get("confidence", 0)
                            bbox = result.get("bbox", [0,0,0,0])
                            print(f"  ✅ Detection | Conf: {conf:.2f} | BBox: {bbox} | Latency: {elapsed}ms")
                        
                        elif task_name == "person_counting":
                            count = result.get("count", 0)
                            conf = result.get("confidence", 0)
                            print(f"  ✅ Counting  | Count: {count} | Avg Conf: {conf:.2f} | Latency: {elapsed}ms")
                            if count > 0 and count <= 3:  # Only show details for small counts
                                for i, det in enumerate(result.get('detections', [])[:3]):
                                    print(f"     Person {i+1}: Conf={det['confidence']}, BBox={det['bbox']}")
                        
                        else:
                            print(f"  ✅ {task_name}: {result}")
                    
                    else:
                        print(f"  ❌ {task_name} Error {r.status_code}: {r.json()}")
                    
                except httpx.TimeoutException:
                    print(f"  ⏱️  {task_name} Timeout - adapter took too long")
                except httpx.ConnectError:
                    print(f"  ❌ {task_name} Connection Error - is adapter running?")
                except Exception as e:
                    print(f"  ❌ {task_name} Error: {e}")
            
            # Wait before next capture (Active mode only)
            if not is_passive:
                time.sleep(interval)
            
    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print(f"🛑 Runner stopped by user")
        print(f"Total frames processed: {frame_count}")
        print(f"{'='*60}\n")
    
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(
        description="Kavach Camera Runner - Run inference tasks on camera feed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runner.py --task person_detection
  python runner.py --task person_counting --interval 1.0
  python runner.py --task person_detection --camera 1
  python runner.py --list-tasks
        """
    )
    
    parser.add_argument(
        "--task",
        type=str,
        help="Task(s) to run. Use comma-separated for multiple (e.g., person_detection or person_detection,person_counting)"
    )
    
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between captures (default: 2.0)"
    )
    
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available tasks from the adapter"
    )
    
    args = parser.parse_args()
    
    # List tasks if requested
    if args.list_tasks:
        print("\n🔍 Fetching available tasks from adapter...\n")
        tasks = get_available_tasks()
        if tasks:
            print("Available tasks:")
            for task in tasks:
                print(f"  - {task}")
            print(f"\nTo run a task: python runner.py --task {tasks[0]}")
        else:
            print("No tasks available or adapter not running.")
        print()
        return
    
    # Validate task is provided
    if not args.task:
        print("\n❌ Error: --task is required")
        print("Use --list-tasks to see available tasks")
        print("Example: python runner.py --task person_detection\n")
        parser.print_help()
        sys.exit(1)
    
    # Verify task is available
    available_tasks = get_available_tasks()
    if available_tasks and args.task not in available_tasks:
        print(f"\n⚠️  Warning: Task '{args.task}' not found in adapter capabilities")
        print(f"Available tasks: {available_tasks}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Run camera
    run_camera(args.task, args.camera, args.interval)


if __name__ == "__main__":
    main()