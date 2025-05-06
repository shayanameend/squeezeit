"""
Huffman Coding implementation for image compression.

This module provides functions to compress images using Huffman Coding.
The implementation directly compresses an image to another image format with
reduced file size while maintaining reasonable quality.
"""

import numpy as np
from PIL import Image
import io
import os
import heapq
from collections import Counter

class HuffmanNode:
    """Node in a Huffman tree."""
    
    def __init__(self, value, freq):
        self.value = value
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

def compress_image(input_path, output_path=None, quality=85):
    """
    Compress an image using Huffman Coding and save it in the same format.
    
    Args:
        input_path: Path to the input image
        output_path: Path to save the compressed image. If None, will use input_path with '_compressed' suffix
        quality: Quality level (0-100), higher means better quality but larger file size
        
    Returns:
        tuple: (output_path, compression_ratio)
    """
    # Determine output path if not provided
    if output_path is None:
        file_name, file_ext = os.path.splitext(input_path)
        output_path = f"{file_name}_compressed{file_ext}"
    
    # Load the image
    original_image = Image.open(input_path)
    original_format = original_image.format if original_image.format else 'JPEG'
    
    # Get original file size
    original_size = os.path.getsize(input_path)
    
    # Convert to numpy array for processing
    img_array = np.array(original_image)
    
    # Process based on image type
    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:  # Color image
        compressed_image = _compress_color_image(img_array, quality_level=quality)
    else:  # Grayscale image
        compressed_image = _compress_grayscale_image(img_array, quality_level=quality)
    
    # Save the compressed image in the same format as the original
    if original_format == 'JPEG' or original_format == 'JPG':
        compressed_image.save(output_path, format=original_format, quality=quality)
    else:
        # For PNG, BMP, etc. use optimize option
        compressed_image.save(output_path, format=original_format, optimize=True)
    
    # Calculate compression ratio
    compressed_size = os.path.getsize(output_path)
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    
    return output_path, compression_ratio

def _compress_color_image(img_array, quality_level=85):
    """
    Compress a color image using Huffman coding techniques.
    
    Args:
        img_array: Numpy array of the image
        quality_level: Quality level (0-100)
        
    Returns:
        PIL.Image: Compressed image
    """
    height, width, channels = img_array.shape
    
    # Apply quantization based on quality level
    quantization_factor = max(1, int((100 - quality_level) / 12))
    quantized = img_array // quantization_factor * quantization_factor
    
    # Process each channel with Huffman coding
    compressed_channels = []
    for c in range(min(channels, 3)):  # Process RGB channels
        channel_data = quantized[:, :, c].flatten()
        
        # Build Huffman tree and codes
        huffman_codes = _build_huffman_codes(channel_data)
        
        # Compress the channel data
        compressed_data = _huffman_compress_data(channel_data, huffman_codes)
        
        # Decompress for reconstruction
        reconstructed = _huffman_decompress_data(compressed_data, huffman_codes, height * width)
        
        # Reshape back to 2D
        reconstructed_channel = np.array(reconstructed).reshape(height, width)
        compressed_channels.append(reconstructed_channel)
    
    # Reconstruct the image from compressed channels
    result = np.zeros((height, width, channels), dtype=np.uint8)
    for c in range(min(channels, 3)):
        result[:, :, c] = compressed_channels[c]
    
    # If there's an alpha channel, preserve it
    if channels == 4:
        result[:, :, 3] = img_array[:, :, 3]
    
    # Create a new image from the processed array
    processed_image = Image.fromarray(result)
    
    return processed_image

def _compress_grayscale_image(img_array, quality_level=85):
    """
    Compress a grayscale image using Huffman coding techniques.
    
    Args:
        img_array: Numpy array of the image
        quality_level: Quality level (0-100)
        
    Returns:
        PIL.Image: Compressed image
    """
    # Ensure the image is 2D
    if len(img_array.shape) == 3:
        img_array = img_array[:, :, 0]
    
    height, width = img_array.shape
    
    # Apply quantization based on quality level
    quantization_factor = max(1, int((100 - quality_level) / 12))
    quantized = img_array // quantization_factor * quantization_factor
    
    # Flatten the image data
    flat_data = quantized.flatten()
    
    # Build Huffman tree and codes
    huffman_codes = _build_huffman_codes(flat_data)
    
    # Compress the data
    compressed_data = _huffman_compress_data(flat_data, huffman_codes)
    
    # Decompress for reconstruction
    reconstructed = _huffman_decompress_data(compressed_data, huffman_codes, height * width)
    
    # Reshape back to 2D
    reconstructed_image = np.array(reconstructed).reshape(height, width)
    
    # Create a new image from the processed array
    processed_image = Image.fromarray(reconstructed_image.astype(np.uint8), mode='L')
    
    return processed_image

def _build_huffman_tree(data):
    """
    Build a Huffman tree from the given data.
    
    Args:
        data: Sequence of values to encode
        
    Returns:
        HuffmanNode: Root of the Huffman tree
    """
    # Count frequency of each value
    counter = Counter(data)
    
    # Create a priority queue (min heap)
    heap = [HuffmanNode(value, freq) for value, freq in counter.items()]
    heapq.heapify(heap)
    
    # Build the Huffman tree
    while len(heap) > 1:
        # Get the two nodes with lowest frequency
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        # Create a new internal node with these two nodes as children
        internal = HuffmanNode(None, left.freq + right.freq)
        internal.left = left
        internal.right = right
        
        # Add the new node to the priority queue
        heapq.heappush(heap, internal)
    
    # Return the root of the Huffman tree
    return heap[0] if heap else None

def _build_huffman_codes(data):
    """
    Build Huffman codes for the given data.
    
    Args:
        data: Sequence of values to encode
        
    Returns:
        dict: Dictionary mapping values to their Huffman codes
    """
    # Build the Huffman tree
    root = _build_huffman_tree(data)
    
    if not root:
        return {}
    
    # Build codes recursively
    codes = {}
    _build_codes_recursive(root, "", codes)
    
    return codes

def _build_codes_recursive(node, code, codes):
    """
    Recursively build Huffman codes from a Huffman tree.
    
    Args:
        node: Current node in the Huffman tree
        code: Current code string (used in recursion)
        codes: Dictionary to store the codes
    """
    # If this is a leaf node, store the code
    if node.value is not None:
        codes[node.value] = code
    else:
        # Traverse left (add '0')
        if node.left:
            _build_codes_recursive(node.left, code + '0', codes)
        
        # Traverse right (add '1')
        if node.right:
            _build_codes_recursive(node.right, code + '1', codes)

def _huffman_compress_data(data, codes):
    """
    Compress data using Huffman codes.
    
    Args:
        data: Data to compress
        codes: Huffman codes dictionary
        
    Returns:
        tuple: (encoded_data, code_table)
    """
    # Encode the data
    encoded = ''.join(codes.get(value, '') for value in data)
    
    # Convert binary string to bytes for storage
    # Pad to make length a multiple of 8
    padding = 8 - (len(encoded) % 8) if len(encoded) % 8 != 0 else 0
    encoded += '0' * padding
    
    # Convert to bytes
    encoded_bytes = bytearray()
    for i in range(0, len(encoded), 8):
        byte = int(encoded[i:i+8], 2)
        encoded_bytes.append(byte)
    
    # Return the compressed data and the code table for decompression
    return (encoded_bytes, codes, padding)

def _huffman_decompress_data(compressed_data, codes, original_length):
    """
    Decompress Huffman-encoded data.
    
    Args:
        compressed_data: Tuple of (encoded_bytes, codes, padding)
        codes: Huffman codes dictionary
        original_length: Original length of the data
        
    Returns:
        list: Decompressed data
    """
    encoded_bytes, codes, padding = compressed_data
    
    # Convert bytes back to binary string
    binary_data = ''
    for byte in encoded_bytes:
        binary_data += format(byte, '08b')
    
    # Remove padding
    binary_data = binary_data[:-padding] if padding else binary_data
    
    # Create reverse mapping (code -> value)
    reverse_codes = {code: value for value, code in codes.items()}
    
    # Decode the data
    decoded = []
    current_code = ''
    
    for bit in binary_data:
        current_code += bit
        if current_code in reverse_codes:
            decoded.append(reverse_codes[current_code])
            current_code = ''
            
            # Stop if we've reached the original length
            if len(decoded) >= original_length:
                break
    
    # Ensure we have the correct length
    if len(decoded) < original_length:
        # Pad with the last value if necessary
        last_value = decoded[-1] if decoded else 0
        decoded.extend([last_value] * (original_length - len(decoded)))
    
    return decoded

def get_supported_extensions():
    """
    Get the list of file extensions supported by this compression method.
    
    Returns:
        list: List of supported file extensions
    """
    return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
