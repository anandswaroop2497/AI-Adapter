"""
Image processing utilities for the AI adapter.
"""
import os
import cv2
from fastapi import HTTPException
from ..config import BASE_FRAMES_DIR


def load_image_from_uri(uri: str):
    """
    Load an image from a Kavach URI.
    
    Args:
        uri: Kavach URI in format kavach://frames/<camera_id>/<filename>
        
    Returns:
        Loaded image as numpy array
        
    Raises:
        HTTPException: If frame not found or invalid
    """
    # Kavach URI format: kavach://frames/<camera_id>/<filename>
    # Strip "kavach://frames/" and join with local frames dir
    relative_path = uri.replace("kavach://frames/", "").replace("/", os.sep)
    frame_path = os.path.join(BASE_FRAMES_DIR, relative_path)
    
    if not os.path.exists(frame_path):
        raise HTTPException(status_code=404, detail=f"Frame not found: {frame_path}")
    
    img = cv2.imread(frame_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    
    return img


def validate_image(img):
    """
    Validate that an image is properly loaded.
    
    Args:
        img: Image to validate
        
    Returns:
        True if valid
        
    Raises:
        HTTPException: If image is invalid
    """
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    return True
