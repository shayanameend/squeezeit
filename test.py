import os
import time
import argparse
import matplotlib.pyplot as plt

from image_compression import rle, huffman

def test_compression(image_path, output_dir=None, qualities=None):
    """
    Test both compression algorithms on the same image with different quality settings.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save output images (optional)
        qualities: List of quality values to test (default: [10, 30, 50, 70, 90])
    
    Returns:
        Dictionary with test results
    """
    if qualities is None:
        qualities = [10, 30, 50, 70, 90]
    
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get original image info
    file_name, file_ext = os.path.splitext(os.path.basename(image_path))
    
    results = {
        'qualities': qualities,
        'rle': {
            'compression_ratios': [],
            'times': []
        },
        'huffman': {
            'compression_ratios': [],
            'times': []
        }
    }
    
    # Test each quality level
    for quality in qualities:
        # RLE compression
        rle_output_path = os.path.join(output_dir, f"{file_name}_rle_q{quality}{file_ext}")
        
        start_time = time.time()
        _, rle_ratio = rle.compress_image(image_path, rle_output_path, quality)
        rle_time = time.time() - start_time
        
        results['rle']['compression_ratios'].append(rle_ratio)
        results['rle']['times'].append(rle_time)
        
        # Huffman compression
        huffman_output_path = os.path.join(output_dir, f"{file_name}_huffman_q{quality}{file_ext}")
        
        start_time = time.time()
        _, huffman_ratio = huffman.compress_image(image_path, huffman_output_path, quality)
        huffman_time = time.time() - start_time
        
        results['huffman']['compression_ratios'].append(huffman_ratio)
        results['huffman']['times'].append(huffman_time)
        
        print(f"Quality {quality}:")
        print(f"  RLE: Ratio = {rle_ratio:.2f}x, Time = {rle_time:.4f}s")
        print(f"  Huffman: Ratio = {huffman_ratio:.2f}x, Time = {huffman_time:.4f}s")
    
    return results

def plot_results(results, output_path=None):
    """
    Plot the compression results.
    
    Args:
        results: Dictionary with test results
        output_path: Path to save the plot (optional)
    """
    qualities = results['qualities']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot compression ratios
    ax1.plot(qualities, results['rle']['compression_ratios'], 'b-o', label='RLE')
    ax1.plot(qualities, results['huffman']['compression_ratios'], 'r-o', label='Huffman')
    ax1.set_xlabel('Quality')
    ax1.set_ylabel('Compression Ratio')
    ax1.set_title('Compression Ratio vs. Quality')
    ax1.legend()
    ax1.grid(True)
    
    # Plot compression times
    ax2.plot(qualities, results['rle']['times'], 'b-o', label='RLE')
    ax2.plot(qualities, results['huffman']['times'], 'r-o', label='Huffman')
    ax2.set_xlabel('Quality')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Compression Time vs. Quality')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Test RLE and Huffman compression algorithms')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--output-dir', help='Directory to save output images')
    parser.add_argument('--plot', help='Path to save the plot')
    parser.add_argument('--qualities', type=int, nargs='+', default=[10, 30, 50, 70, 90],
                        help='Quality values to test (default: 10 30 50 70 90)')
    
    args = parser.parse_args()
    
    results = test_compression(args.image_path, args.output_dir, args.qualities)
    plot_results(results, args.plot)

if __name__ == '__main__':
    main()
