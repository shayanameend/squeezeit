"""
File utility functions for the Squeezeit Compression Tool.

This module provides utility functions for file operations.
"""

from PIL import Image
import cv2
import os
import tempfile

def get_file_size(file_path):
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        int: Size of the file in bytes
    """
    return os.path.getsize(file_path)

def format_size(size_bytes):
    """
    Format file size in a human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Formatted size string (e.g., "1.23 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0 or unit == 'GB':
            break
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} {unit}"

def load_image(file_path):
    """
    Load an image from a file.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        PIL.Image: Loaded image
    """
    return Image.open(file_path)

def save_image(image, file_path):
    """
    Save an image to a file.
    
    Args:
        image: PIL Image object
        file_path: Path to save the image
    """
    image.save(file_path)

def load_video(file_path):
    """
    Load a video from a file.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        tuple: (frames, fps)
            - frames: List of numpy arrays representing video frames
            - fps: Frames per second of the video
    """
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    return frames, fps

def save_video(frames, fps, file_path):
    """
    Save a video to a file.
    
    Args:
        frames: List of numpy arrays representing video frames
        fps: Frames per second
        file_path: Path to save the video
    """
    if not frames:
        raise ValueError("No frames to save")
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()

def save_compressed_data(data, file_path):
    """
    Save compressed binary data to a file.
    
    Args:
        data: Compressed binary data
        file_path: Path to save the data
    """
    with open(file_path, 'wb') as f:
        f.write(data)

def load_compressed_data(file_path):
    """
    Load compressed binary data from a file.
    
    Args:
        file_path: Path to the compressed data file
        
    Returns:
        bytes: Compressed binary data
    """
    with open(file_path, 'rb') as f:
        return f.read()

def get_temp_file_path(extension):
    """
    Get a temporary file path with the specified extension.
    
    Args:
        extension: File extension (e.g., '.jpg', '.mp4')
        
    Returns:
        str: Temporary file path
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=extension, delete=False)
    temp_file.close()
    return temp_file.name
