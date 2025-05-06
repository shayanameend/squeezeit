"""
Main window for the Squeezeit Compression Tool.

This module provides the main GUI window for the application.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QTabWidget, QSizePolicy, QProgressBar, QMessageBox, QGroupBox, QRadioButton,
    QButtonGroup, QSlider
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from PIL import Image, ImageQt
import os
import cv2
import numpy as np

from image_compression import rle, huffman
from video_compression import delta, motion

class CompressionThread(QThread):
    """Thread for running compression operations."""

    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(tuple)
    error_signal = pyqtSignal(str)

    def __init__(self, file_type, algorithm, input_path, output_path, quality=85):
        """Initialize the thread."""
        super().__init__()
        self.file_type = file_type  # 'image' or 'video'
        self.algorithm = algorithm
        self.input_path = input_path
        self.output_path = output_path
        self.quality = quality

    def run(self):
        """Run the compression operation."""
        try:
            self.progress_signal.emit(10)

            if self.file_type == 'image':
                # Compress image
                if self.algorithm == 'rle':
                    output_path, ratio = rle.compress_image(self.input_path, self.output_path, self.quality)
                elif self.algorithm == 'huffman':
                    output_path, ratio = huffman.compress_image(self.input_path, self.output_path, self.quality)
                else:
                    raise ValueError(f"Unknown image algorithm: {self.algorithm}")

                self.progress_signal.emit(90)

                # Return the compressed image path and ratio
                self.finished_signal.emit((output_path, ratio))

            elif self.file_type == 'video':
                # Compress video
                if self.algorithm == 'delta':
                    output_path, ratio = delta.compress_video(self.input_path, self.output_path, self.quality)
                elif self.algorithm == 'motion':
                    output_path, ratio = motion.compress_video(self.input_path, self.output_path, self.quality)
                else:
                    raise ValueError(f"Unknown video algorithm: {self.algorithm}")

                self.progress_signal.emit(90)

                # Return the compressed video path and ratio
                self.finished_signal.emit((output_path, ratio))

            else:
                raise ValueError(f"Unknown file type: {self.file_type}")

            self.progress_signal.emit(100)

        except Exception as e:
            self.error_signal.emit(str(e))

class MainWindow(QMainWindow):
    """Main window for the Squeezeit Compression Tool."""

    def __init__(self):
        """Initialize the main window."""
        super().__init__()

        self.setWindowTitle("Squeezeit Compression Tool")
        self.setMinimumSize(800, 600)

        # Initialize instance variables
        self.input_file_path = None
        self.output_file_path = None
        self.file_type = None  # 'image' or 'video'
        self.current_algorithm = None
        self.quality = 85  # Default quality level

        # Set up the UI
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # Create tabs for image and video compression
        tabs = QTabWidget()
        image_tab = QWidget()
        video_tab = QWidget()

        tabs.addTab(image_tab, "Image Compression")
        tabs.addTab(video_tab, "Video Compression")

        # Set up image tab
        image_layout = QVBoxLayout(image_tab)

        # File selection section
        file_group = QGroupBox("Input Image")
        file_layout = QHBoxLayout(file_group)

        self.image_file_label = QLabel("No file selected")
        file_layout.addWidget(self.image_file_label)

        select_file_button = QPushButton("Select Image")
        select_file_button.clicked.connect(lambda: self.select_file('image'))
        file_layout.addWidget(select_file_button)

        image_layout.addWidget(file_group)

        # Algorithm selection section
        algo_group = QGroupBox("Algorithm")
        algo_layout = QHBoxLayout(algo_group)

        self.image_algo_rle = QRadioButton("Run-Length Encoding (RLE)")
        self.image_algo_huffman = QRadioButton("Huffman Coding")
        self.image_algo_rle.setChecked(True)

        image_algo_group = QButtonGroup(self)
        image_algo_group.addButton(self.image_algo_rle)
        image_algo_group.addButton(self.image_algo_huffman)

        algo_layout.addWidget(self.image_algo_rle)
        algo_layout.addWidget(self.image_algo_huffman)

        image_layout.addWidget(algo_group)

        # Quality slider
        quality_group = QGroupBox("Compression Quality")
        quality_layout = QVBoxLayout(quality_group)

        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setRange(1, 100)
        self.quality_slider.setValue(self.quality)
        self.quality_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.quality_slider.setTickInterval(10)

        self.quality_label = QLabel(f"Quality: {self.quality}%")
        self.quality_slider.valueChanged.connect(self.update_quality)

        quality_layout.addWidget(self.quality_label)
        quality_layout.addWidget(self.quality_slider)

        image_layout.addWidget(quality_group)

        # Action buttons
        action_layout = QHBoxLayout()

        self.compress_image_button = QPushButton("Compress Image")
        self.compress_image_button.clicked.connect(lambda: self.compress('image'))
        self.compress_image_button.setEnabled(False)

        action_layout.addWidget(self.compress_image_button)

        image_layout.addLayout(action_layout)

        # Progress bar
        self.image_progress = QProgressBar()
        self.image_progress.setRange(0, 100)
        self.image_progress.setValue(0)
        image_layout.addWidget(self.image_progress)

        # Image preview section
        preview_layout = QHBoxLayout()

        # Original image
        original_group = QGroupBox("Original Image")
        original_layout = QVBoxLayout(original_group)

        self.original_image_label = QLabel()
        self.original_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        original_layout.addWidget(self.original_image_label)
        self.original_image_size_label = QLabel("Size: N/A")
        original_layout.addWidget(self.original_image_size_label)

        preview_layout.addWidget(original_group)

        # Compressed image
        compressed_group = QGroupBox("Compressed Image")
        compressed_layout = QVBoxLayout(compressed_group)

        self.compressed_image_label = QLabel()
        self.compressed_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.compressed_image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        compressed_layout.addWidget(self.compressed_image_label)
        self.compressed_image_size_label = QLabel("Size: N/A")
        compressed_layout.addWidget(self.compressed_image_size_label)

        preview_layout.addWidget(compressed_group)

        image_layout.addLayout(preview_layout)

        # Set up video tab (similar to image tab)
        video_layout = QVBoxLayout(video_tab)

        # File selection section
        video_file_group = QGroupBox("Input Video")
        video_file_layout = QHBoxLayout(video_file_group)

        self.video_file_label = QLabel("No file selected")
        video_file_layout.addWidget(self.video_file_label)

        select_video_button = QPushButton("Select Video")
        select_video_button.clicked.connect(lambda: self.select_file('video'))
        video_file_layout.addWidget(select_video_button)

        video_layout.addWidget(video_file_group)

        # Algorithm selection section
        video_algo_group_box = QGroupBox("Algorithm")
        video_algo_layout = QHBoxLayout(video_algo_group_box)

        self.video_algo_delta = QRadioButton("Delta Encoding")
        self.video_algo_motion = QRadioButton("Motion Compensation")
        self.video_algo_delta.setChecked(True)

        video_algo_group = QButtonGroup(self)
        video_algo_group.addButton(self.video_algo_delta)
        video_algo_group.addButton(self.video_algo_motion)

        video_algo_layout.addWidget(self.video_algo_delta)
        video_algo_layout.addWidget(self.video_algo_motion)

        video_layout.addWidget(video_algo_group_box)

        # Video quality slider (same as image)
        video_quality_group = QGroupBox("Compression Quality")
        video_quality_layout = QVBoxLayout(video_quality_group)

        self.video_quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_quality_slider.setRange(1, 100)
        self.video_quality_slider.setValue(self.quality)
        self.video_quality_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.video_quality_slider.setTickInterval(10)

        self.video_quality_label = QLabel(f"Quality: {self.quality}%")
        self.video_quality_slider.valueChanged.connect(self.update_video_quality)

        video_quality_layout.addWidget(self.video_quality_label)
        video_quality_layout.addWidget(self.video_quality_slider)

        video_layout.addWidget(video_quality_group)

        # Action buttons
        video_action_layout = QHBoxLayout()

        self.compress_video_button = QPushButton("Compress Video")
        self.compress_video_button.clicked.connect(lambda: self.compress('video'))
        self.compress_video_button.setEnabled(False)

        video_action_layout.addWidget(self.compress_video_button)

        video_layout.addLayout(video_action_layout)

        # Progress bar
        self.video_progress = QProgressBar()
        self.video_progress.setRange(0, 100)
        self.video_progress.setValue(0)
        video_layout.addWidget(self.video_progress)

        # Video preview section (simplified, just showing first frame)
        video_preview_layout = QHBoxLayout()

        # Original video
        original_video_group = QGroupBox("Original Video")
        original_video_layout = QVBoxLayout(original_video_group)

        self.original_video_label = QLabel()
        self.original_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        original_video_layout.addWidget(self.original_video_label)
        self.original_video_size_label = QLabel("Size: N/A")
        original_video_layout.addWidget(self.original_video_size_label)

        video_preview_layout.addWidget(original_video_group)

        # Compressed video
        compressed_video_group = QGroupBox("Compressed Video")
        compressed_video_layout = QVBoxLayout(compressed_video_group)

        self.compressed_video_label = QLabel()
        self.compressed_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.compressed_video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        compressed_video_layout.addWidget(self.compressed_video_label)
        self.compressed_video_size_label = QLabel("Size: N/A")
        compressed_video_layout.addWidget(self.compressed_video_size_label)

        video_preview_layout.addWidget(compressed_video_group)

        video_layout.addLayout(video_preview_layout)

        # Add tabs to main layout
        main_layout.addWidget(tabs)

        # Set central widget
        self.setCentralWidget(central_widget)

    def update_quality(self, value):
        """Update the quality value and label."""
        self.quality = value
        self.quality_label.setText(f"Quality: {value}%")
        self.video_quality_slider.setValue(value)
        self.video_quality_label.setText(f"Quality: {value}%")

    def update_video_quality(self, value):
        """Update the video quality value and label."""
        self.quality = value
        self.video_quality_label.setText(f"Quality: {value}%")
        self.quality_slider.setValue(value)
        self.quality_label.setText(f"Quality: {value}%")

    def select_file(self, file_type):
        """
        Open a file dialog to select an input file.

        Args:
            file_type: Type of file to select ('image' or 'video')
        """
        if file_type == 'image':
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.gif)"
            )
        else:  # video
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
            )

        if file_path:
            self.input_file_path = file_path
            self.file_type = file_type

            # Update UI
            if file_type == 'image':
                self.image_file_label.setText(os.path.basename(file_path))
                self.compress_image_button.setEnabled(True)

                # Display original image
                self.display_image(file_path, self.original_image_label)

                # Update size label
                size = os.path.getsize(file_path)
                self.original_image_size_label.setText(f"Size: {self.format_size(size)}")

                # Clear compressed image
                self.compressed_image_label.clear()
                self.compressed_image_size_label.setText("Size: N/A")

            else:  # video
                self.video_file_label.setText(os.path.basename(file_path))
                self.compress_video_button.setEnabled(True)

                # Display first frame of original video
                self.display_video_frame(file_path, self.original_video_label)

                # Update size label
                size = os.path.getsize(file_path)
                self.original_video_size_label.setText(f"Size: {self.format_size(size)}")

                # Clear compressed video
                self.compressed_video_label.clear()
                self.compressed_video_size_label.setText("Size: N/A")

    def compress(self, file_type):
        """
        Compress the selected file.

        Args:
            file_type: Type of file to compress ('image' or 'video')
        """
        if not self.input_file_path:
            return

        # Determine algorithm
        if file_type == 'image':
            if self.image_algo_rle.isChecked():
                algorithm = 'rle'
            else:
                algorithm = 'huffman'
        else:  # video
            if self.video_algo_delta.isChecked():
                algorithm = 'delta'
            else:
                algorithm = 'motion'

        self.current_algorithm = algorithm

        # Get output file path
        if file_type == 'image':
            # Get the original file extension
            _, ext = os.path.splitext(self.input_file_path)

            # Use the same extension for output
            output_path, _ = QFileDialog.getSaveFileName(
                self, "Save Compressed Image", "", f"Image Files (*{ext})"
            )
        else:  # video
            # Get the original file extension
            _, ext = os.path.splitext(self.input_file_path)

            # Use the same extension for output
            output_path, _ = QFileDialog.getSaveFileName(
                self, "Save Compressed Video", "", f"Video Files (*{ext})"
            )

        if not output_path:
            return

        # Add extension if not present
        if not output_path.endswith(ext):
            output_path += ext

        self.output_file_path = output_path

        # Disable buttons during compression
        if file_type == 'image':
            self.compress_image_button.setEnabled(False)
            progress_bar = self.image_progress
        else:  # video
            self.compress_video_button.setEnabled(False)
            progress_bar = self.video_progress

        # Create and start compression thread
        self.compression_thread = CompressionThread(
            file_type, algorithm, self.input_file_path, output_path, self.quality
        )
        self.compression_thread.progress_signal.connect(progress_bar.setValue)
        self.compression_thread.finished_signal.connect(
            lambda result: self.compression_finished(result, file_type)
        )
        self.compression_thread.error_signal.connect(self.show_error)
        self.compression_thread.start()

    def compression_finished(self, result, file_type):
        """
        Handle completion of compression operation.

        Args:
            result: Tuple of (output_path, compression_ratio)
            file_type: Type of file that was compressed ('image' or 'video')
        """
        output_path, compression_ratio = result

        # Re-enable buttons
        if file_type == 'image':
            self.compress_image_button.setEnabled(True)

            # Display compressed image
            self.display_image(output_path, self.compressed_image_label)

            # Update size label
            size = os.path.getsize(output_path)
            self.compressed_image_size_label.setText(
                f"Size: {self.format_size(size)} (Ratio: {compression_ratio:.2f}x)"
            )

        else:  # video
            self.compress_video_button.setEnabled(True)

            # Display first frame of compressed video
            self.display_video_frame(output_path, self.compressed_video_label)

            # Update size label
            size = os.path.getsize(output_path)
            self.compressed_video_size_label.setText(
                f"Size: {self.format_size(size)} (Ratio: {compression_ratio:.2f}x)"
            )

        # Show success message
        QMessageBox.information(
            self,
            "Compression Complete",
            f"File compressed successfully!\nCompression ratio: {compression_ratio:.2f}x"
        )

    def show_error(self, error_message):
        """
        Show an error message.

        Args:
            error_message: Error message to display
        """
        QMessageBox.critical(self, "Error", error_message)

        # Re-enable buttons
        if self.file_type == 'image':
            self.compress_image_button.setEnabled(True)
        else:  # video
            self.compress_video_button.setEnabled(True)

    def display_image(self, image_path, label):
        """
        Display an image in a QLabel.

        Args:
            image_path: Path to the image file
            label: QLabel to display the image in
        """
        try:
            # Load image
            image = Image.open(image_path)

            # Convert to QPixmap
            qimage = ImageQt.ImageQt(image)
            pixmap = QPixmap.fromImage(qimage)

            # Scale pixmap to fit label while maintaining aspect ratio
            pixmap = pixmap.scaled(
                label.width(), label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            # Set pixmap to label
            label.setPixmap(pixmap)

        except Exception as e:
            self.show_error(f"Error displaying image: {str(e)}")

    def display_video_frame(self, video_path, label):
        """
        Display the first frame of a video in a QLabel.

        Args:
            video_path: Path to the video file
            label: QLabel to display the frame in
        """
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)

            # Read first frame
            ret, frame = cap.read()

            if ret:
                # Convert from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Create QImage from frame
                height, width, channels = frame_rgb.shape
                bytes_per_line = channels * width
                qimage = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

                # Convert to QPixmap
                pixmap = QPixmap.fromImage(qimage)

                # Scale pixmap to fit label while maintaining aspect ratio
                pixmap = pixmap.scaled(
                    label.width(), label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )

                # Set pixmap to label
                label.setPixmap(pixmap)

            # Release video capture
            cap.release()

        except Exception as e:
            self.show_error(f"Error displaying video frame: {str(e)}")

    def format_size(self, size_bytes):
        """
        Format file size in human-readable format.

        Args:
            size_bytes: Size in bytes

        Returns:
            str: Formatted size string
        """
        # Define size units
        units = ['B', 'KB', 'MB', 'GB', 'TB']

        # Calculate unit index and size
        unit_index = 0
        size = float(size_bytes)

        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1

        # Format size string
        if unit_index == 0:
            return f"{int(size)} {units[unit_index]}"
        else:
            return f"{size:.2f} {units[unit_index]}"
