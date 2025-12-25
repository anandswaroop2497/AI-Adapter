import os
import sys

def check_file(path, description):
    if os.path.exists(path):
        print(f"[OK] {description} found: {path}")
        return True
    else:
        print(f"[FAIL] {description} NOT found: {path}")
        return False

def check_imports():
    print("Checking imports...")
    try:
        import fastapi
        import onnxruntime
        import cv2
        import numpy
        import ultralytics
        print("[OK] All imports successful.")
        return True
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False

def verify_main_requirements():
    print("\nVerifying adapter/main.py requirements...")
    # Check for model file
    model_filename = "yolov8n.onnx"
    # main.py expects it in the current directory (or relative to execution)
    # The list_dir showed it's NOT there, but yolov8n.pt IS there.
    
    if check_file(model_filename, "ONNX Model"):
        pass
    else:
        print(f"    Available files: {os.listdir('.')}")
        if os.path.exists("yolov8n.pt"):
            print("[INFO] Found yolov8n.pt. You might need to export it to ONNX.")

def main():
    if not check_imports():
        sys.exit(1)
    
    verify_main_requirements()
    
    print("\nVerification script finished.")

if __name__ == "__main__":
    main()
