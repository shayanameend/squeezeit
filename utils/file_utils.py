from PIL import Image
import cv2
import os
import tempfile

def get_file_size(file_path):
    return os.path.getsize(file_path)

def format_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0 or unit == 'GB':
            break
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} {unit}"

def load_image(file_path):
    return Image.open(file_path)

def save_image(image, file_path):
    image.save(file_path)

def load_video(file_path):
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    return frames, fps

def save_video(frames, fps, file_path):
    if not frames:
        raise ValueError("No frames to save")
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()

def save_compressed_data(data, file_path):
    with open(file_path, 'wb') as f:
        f.write(data)

def load_compressed_data(file_path):
    with open(file_path, 'rb') as f:
        return f.read()

def get_temp_file_path(extension):
    temp_file = tempfile.NamedTemporaryFile(suffix=extension, delete=False)
    temp_file.close()
    return temp_file.name
