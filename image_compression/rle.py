import numpy as np
from PIL import Image
import os

def compress_image(input_path, output_path=None, quality=85):
    if output_path is None:
        file_name, file_ext = os.path.splitext(input_path)
        output_path = f"{file_name}_compressed{file_ext}"

    original_image = Image.open(input_path)
    original_format = original_image.format if original_image.format else 'JPEG'

    original_size = os.path.getsize(input_path)

    img_array = np.array(original_image)

    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        compressed_image = _compress_color_image(img_array, quality_level=quality)
    else:
        compressed_image = _compress_grayscale_image(img_array, quality_level=quality)

    if original_format == 'JPEG' or original_format == 'JPG':
        compressed_image.save(output_path, format=original_format, quality=quality)
    else:
        compressed_image.save(output_path, format=original_format, optimize=True)

    compressed_size = os.path.getsize(output_path)
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0

    return output_path, compression_ratio

def _compress_color_image(img_array, quality_level=85):
    height, width, channels = img_array.shape

    quantization_factor = max(1, int((100 - quality_level) / 12))
    quantized = img_array // quantization_factor * quantization_factor

    result = np.zeros_like(img_array)

    for c in range(min(channels, 3)):
        channel_data = quantized[:, :, c]

        if quality_level < 90:
            smoothed = np.zeros_like(channel_data)
            for i in range(1, height-1):
                for j in range(1, width-1):
                    smoothed[i, j] = np.mean(channel_data[i-1:i+2, j-1:j+2])

            blend_factor = max(0, min(1, (90 - quality_level) / 90))
            channel_data = (channel_data * (1 - blend_factor) + smoothed * blend_factor).astype(np.uint8)

        compressed_rows = []
        for row in channel_data:
            compressed_row = _rle_compress_row(row)
            compressed_rows.append(compressed_row)

        reconstructed = np.zeros((height, width), dtype=np.uint8)
        for i in range(height):
            reconstructed[i, :] = compressed_rows[i][:width]

        result[:, :, c] = reconstructed

    if channels == 4:
        result[:, :, 3] = img_array[:, :, 3]

    processed_image = Image.fromarray(result)

    return processed_image

def _compress_grayscale_image(img_array, quality_level=85):
    if len(img_array.shape) == 3:
        img_array = img_array[:, :, 0]

    height, width = img_array.shape

    quantization_factor = max(1, int((100 - quality_level) / 12))
    quantized = img_array // quantization_factor * quantization_factor

    if quality_level < 90:
        smoothed = np.zeros_like(quantized)
        for i in range(1, height-1):
            for j in range(1, width-1):
                smoothed[i, j] = np.mean(quantized[i-1:i+2, j-1:j+2])

        blend_factor = max(0, min(1, (90 - quality_level) / 90))
        quantized = (quantized * (1 - blend_factor) + smoothed * blend_factor).astype(np.uint8)

    compressed_rows = []
    for row in quantized:
        compressed_row = _rle_compress_row(row)
        compressed_rows.append(compressed_row)

    reconstructed = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        reconstructed[i, :] = compressed_rows[i][:width]

    processed_image = Image.fromarray(reconstructed, mode='L')

    return processed_image

def _rle_compress_row(row):
    if len(row) == 0:
        return []
    result = np.copy(row)

    i = 1
    while i < len(row) - 1:
        if abs(int(row[i-1]) - int(row[i])) <= 8 and abs(int(row[i+1]) - int(row[i])) <= 8:
            result[i] = (int(row[i-1]) + int(row[i]) + int(row[i+1])) // 3
        i += 1

    return result

def get_supported_extensions():
    return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
