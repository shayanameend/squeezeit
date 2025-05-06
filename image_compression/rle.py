import os
import numpy as np
from PIL import Image

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

        flat_data = channel_data.flatten()

        compressed_data = _rle_compress_data(flat_data)

        decompressed_data = _rle_decompress_data(compressed_data, height * width)

        reconstructed_channel = np.array(decompressed_data).reshape(height, width)
        result[:, :, c] = reconstructed_channel

    if channels == 4:
        result[:, :, 3] = img_array[:, :, 3]

    processed_image = Image.fromarray(result.astype(np.uint8))

    return processed_image

def _compress_grayscale_image(img_array, quality_level=85):
    if len(img_array.shape) == 3:
        img_array = img_array[:, :, 0]

    height, width = img_array.shape

    quantization_factor = max(1, int((100 - quality_level) / 12))
    quantized = img_array // quantization_factor * quantization_factor

    flat_data = quantized.flatten()

    compressed_data = _rle_compress_data(flat_data)

    decompressed_data = _rle_decompress_data(compressed_data, height * width)

    reconstructed_image = np.array(decompressed_data).reshape(height, width)

    processed_image = Image.fromarray(reconstructed_image.astype(np.uint8), mode='L')

    return processed_image

def _rle_compress_data(data):
    if len(data) == 0:
        return []

    compressed = []
    current_value = data[0]
    count = 1

    for i in range(1, len(data)):
        if data[i] == current_value:
            count += 1
        else:
            compressed.append((current_value, count))
            current_value = data[i]
            count = 1

    compressed.append((current_value, count))

    return compressed

def _rle_decompress_data(compressed_data, original_length):
    decompressed = []

    for value, count in compressed_data:
        decompressed.extend([value] * count)

    if len(decompressed) < original_length:
        last_value = decompressed[-1] if decompressed else 0
        decompressed.extend([last_value] * (original_length - len(decompressed)))
    elif len(decompressed) > original_length:
        decompressed = decompressed[:original_length]

    return decompressed

def get_supported_extensions():
    return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
