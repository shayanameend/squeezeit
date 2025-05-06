import numpy as np
from PIL import Image
import os
import heapq
from collections import Counter

class HuffmanNode:    
    def __init__(self, value, freq):
        self.value = value
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

def compress_image(input_path, output_path=None, quality=85):
    if output_path is None:
        file_name, file_ext = os.path.splitext(input_path)
        output_path = f"{file_name}_compressed{file_ext}"
    
    original_image = Image.open(input_path)
    original_format = original_image.format if original_image.format else 'JPEG'
    
    original_size = os.path.getsize(input_path)
    
    img_array = np.array(original_image)
    
    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:  # Color image
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
    
    compressed_channels = []
    for c in range(min(channels, 3)):
        channel_data = quantized[:, :, c].flatten()
        
        huffman_codes = _build_huffman_codes(channel_data)
        
        compressed_data = _huffman_compress_data(channel_data, huffman_codes)
        
        reconstructed = _huffman_decompress_data(compressed_data, huffman_codes, height * width)
        
        reconstructed_channel = np.array(reconstructed).reshape(height, width)
        compressed_channels.append(reconstructed_channel)
    
    result = np.zeros((height, width, channels), dtype=np.uint8)
    for c in range(min(channels, 3)):
        result[:, :, c] = compressed_channels[c]
    
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
    
    flat_data = quantized.flatten()
    
    huffman_codes = _build_huffman_codes(flat_data)
    
    compressed_data = _huffman_compress_data(flat_data, huffman_codes)
    
    reconstructed = _huffman_decompress_data(compressed_data, huffman_codes, height * width)
    
    reconstructed_image = np.array(reconstructed).reshape(height, width)
    
    processed_image = Image.fromarray(reconstructed_image.astype(np.uint8), mode='L')
    
    return processed_image

def _build_huffman_tree(data):
    counter = Counter(data)
    
    heap = [HuffmanNode(value, freq) for value, freq in counter.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        internal = HuffmanNode(None, left.freq + right.freq)
        internal.left = left
        internal.right = right
        
        heapq.heappush(heap, internal)
    
    return heap[0] if heap else None

def _build_huffman_codes(data):
    root = _build_huffman_tree(data)
    
    if not root:
        return {}
    
    codes = {}
    _build_codes_recursive(root, "", codes)
    
    return codes

def _build_codes_recursive(node, code, codes):
    if node.value is not None:
        codes[node.value] = code
    else:
        if node.left:
            _build_codes_recursive(node.left, code + '0', codes)
        
        if node.right:
            _build_codes_recursive(node.right, code + '1', codes)

def _huffman_compress_data(data, codes):
    encoded = ''.join(codes.get(value, '') for value in data)
    
    padding = 8 - (len(encoded) % 8) if len(encoded) % 8 != 0 else 0
    encoded += '0' * padding
    
    encoded_bytes = bytearray()
    for i in range(0, len(encoded), 8):
        byte = int(encoded[i:i+8], 2)
        encoded_bytes.append(byte)
    
    return (encoded_bytes, codes, padding)

def _huffman_decompress_data(compressed_data, codes, original_length):
    encoded_bytes, codes, padding = compressed_data
    
    binary_data = ''
    for byte in encoded_bytes:
        binary_data += format(byte, '08b')
    
    binary_data = binary_data[:-padding] if padding else binary_data
    
    reverse_codes = {code: value for value, code in codes.items()}
    
    decoded = []
    current_code = ''
    
    for bit in binary_data:
        current_code += bit
        if current_code in reverse_codes:
            decoded.append(reverse_codes[current_code])
            current_code = ''
            
            if len(decoded) >= original_length:
                break
    
    if len(decoded) < original_length:
        last_value = decoded[-1] if decoded else 0
        decoded.extend([last_value] * (original_length - len(decoded)))
    
    return decoded

def get_supported_extensions():
    return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
