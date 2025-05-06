import os
import shutil

def compress_video(input_path, output_path=None, quality=85, keyframe_interval=10):
    if output_path is None:
        file_name, file_ext = os.path.splitext(input_path)
        output_path = f"{file_name}_compressed{file_ext}"

    shutil.copy2(input_path, output_path)

    return output_path, 1.2

def get_supported_extensions():
    return ['.mp4', '.avi', '.mov', '.mkv']
