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

from image_compression import rle, huffman
from video_compression import delta, motion

class CompressionThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(tuple)
    error_signal = pyqtSignal(str)

    def __init__(self, file_type, algorithm, input_path, output_path, quality=85):
        super().__init__()
        self.file_type = file_type
        self.algorithm = algorithm
        self.input_path = input_path
        self.output_path = output_path
        self.quality = quality

    def run(self):
        try:
            self.progress_signal.emit(10)

            if self.file_type == 'image':
                if self.algorithm == 'rle':
                    output_path, ratio = rle.compress_image(self.input_path, self.output_path, self.quality)
                elif self.algorithm == 'huffman':
                    output_path, ratio = huffman.compress_image(self.input_path, self.output_path, self.quality)
                else:
                    raise ValueError(f"Unknown image algorithm: {self.algorithm}")

                self.progress_signal.emit(90)

                self.finished_signal.emit((output_path, ratio))

            elif self.file_type == 'video':
                if self.algorithm == 'delta':
                    output_path, ratio = delta.compress_video(self.input_path, self.output_path, self.quality)
                elif self.algorithm == 'motion':
                    output_path, ratio = motion.compress_video(self.input_path, self.output_path, self.quality)
                else:
                    raise ValueError(f"Unknown video algorithm: {self.algorithm}")

                self.progress_signal.emit(90)

                self.finished_signal.emit((output_path, ratio))

            else:
                raise ValueError(f"Unknown file type: {self.file_type}")

            self.progress_signal.emit(100)

        except Exception as e:
            self.error_signal.emit(str(e))

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Squeezeit Compression Tool")
        self.setMinimumSize(800, 600)

        self.input_file_path = None
        self.output_file_path = None
        self.file_type = None
        self.current_algorithm = None
        self.quality = 85

        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        tabs = QTabWidget()
        image_tab = QWidget()
        video_tab = QWidget()

        tabs.addTab(image_tab, "Image Compression")
        tabs.addTab(video_tab, "Video Compression")

        image_layout = QVBoxLayout(image_tab)

        file_group = QGroupBox("Input Image")
        file_layout = QHBoxLayout(file_group)

        self.image_file_label = QLabel("No file selected")
        file_layout.addWidget(self.image_file_label)

        select_file_button = QPushButton("Select Image")
        select_file_button.clicked.connect(lambda: self.select_file('image'))
        file_layout.addWidget(select_file_button)

        image_layout.addWidget(file_group)

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

        action_layout = QHBoxLayout()

        self.compress_image_button = QPushButton("Compress Image")
        self.compress_image_button.clicked.connect(lambda: self.compress('image'))
        self.compress_image_button.setEnabled(False)

        action_layout.addWidget(self.compress_image_button)

        image_layout.addLayout(action_layout)

        self.image_progress = QProgressBar()
        self.image_progress.setRange(0, 100)
        self.image_progress.setValue(0)
        image_layout.addWidget(self.image_progress)

        preview_layout = QHBoxLayout()

        original_group = QGroupBox("Original Image")
        original_layout = QVBoxLayout(original_group)

        self.original_image_label = QLabel()
        self.original_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        original_layout.addWidget(self.original_image_label)
        self.original_image_size_label = QLabel("Size: N/A")
        original_layout.addWidget(self.original_image_size_label)

        preview_layout.addWidget(original_group)

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

        video_layout = QVBoxLayout(video_tab)

        video_file_group = QGroupBox("Input Video")
        video_file_layout = QHBoxLayout(video_file_group)

        self.video_file_label = QLabel("No file selected")
        video_file_layout.addWidget(self.video_file_label)

        select_video_button = QPushButton("Select Video")
        select_video_button.clicked.connect(lambda: self.select_file('video'))
        video_file_layout.addWidget(select_video_button)

        video_layout.addWidget(video_file_group)

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

        video_action_layout = QHBoxLayout()

        self.compress_video_button = QPushButton("Compress Video")
        self.compress_video_button.clicked.connect(lambda: self.compress('video'))
        self.compress_video_button.setEnabled(False)

        video_action_layout.addWidget(self.compress_video_button)

        video_layout.addLayout(video_action_layout)

        self.video_progress = QProgressBar()
        self.video_progress.setRange(0, 100)
        self.video_progress.setValue(0)
        video_layout.addWidget(self.video_progress)

        video_preview_layout = QHBoxLayout()

        original_video_group = QGroupBox("Original Video")
        original_video_layout = QVBoxLayout(original_video_group)

        self.original_video_label = QLabel()
        self.original_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        original_video_layout.addWidget(self.original_video_label)
        self.original_video_size_label = QLabel("Size: N/A")
        original_video_layout.addWidget(self.original_video_size_label)

        video_preview_layout.addWidget(original_video_group)

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

        main_layout.addWidget(tabs)

        self.setCentralWidget(central_widget)

    def update_quality(self, value):
        self.quality = value
        self.quality_label.setText(f"Quality: {value}%")
        self.video_quality_slider.setValue(value)
        self.video_quality_label.setText(f"Quality: {value}%")

    def update_video_quality(self, value):
        self.quality = value
        self.video_quality_label.setText(f"Quality: {value}%")
        self.quality_slider.setValue(value)
        self.quality_label.setText(f"Quality: {value}%")

    def select_file(self, file_type):
        if file_type == 'image':
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.gif)"
            )

        elif file_type == 'video':
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
            )

        if file_path:
            self.input_file_path = file_path
            self.file_type = file_type

            if file_type == 'image':
                self.image_file_label.setText(os.path.basename(file_path))
                self.compress_image_button.setEnabled(True)

                self.display_image(file_path, self.original_image_label)

                size = os.path.getsize(file_path)
                self.original_image_size_label.setText(f"Size: {self.format_size(size)}")

                self.compressed_image_label.clear()
                self.compressed_image_size_label.setText("Size: N/A")

            elif file_type == 'video':
                self.video_file_label.setText(os.path.basename(file_path))
                self.compress_video_button.setEnabled(True)

                self.display_video_frame(file_path, self.original_video_label)

                size = os.path.getsize(file_path)
                self.original_video_size_label.setText(f"Size: {self.format_size(size)}")

                self.compressed_video_label.clear()
                self.compressed_video_size_label.setText("Size: N/A")

    def compress(self, file_type):
        if not self.input_file_path:
            return

        if file_type == 'image':
            if self.image_algo_rle.isChecked():
                algorithm = 'rle'
            else:
                algorithm = 'huffman'

        elif file_type == 'video':
            if self.video_algo_delta.isChecked():
                algorithm = 'delta'
            else:
                algorithm = 'motion'

        self.current_algorithm = algorithm

        if file_type == 'image':
            _, ext = os.path.splitext(self.input_file_path)

            output_path, _ = QFileDialog.getSaveFileName(
                self, "Save Compressed Image", "", f"Image Files (*{ext})"
            )

        elif file_type == 'video':
            _, ext = os.path.splitext(self.input_file_path)

            output_path, _ = QFileDialog.getSaveFileName(
                self, "Save Compressed Video", "", f"Video Files (*{ext})"
            )

        if not output_path:
            return

        # Add extension if not present
        if not output_path.endswith(ext):
            output_path += ext

        self.output_file_path = output_path

        if file_type == 'image':
            self.compress_image_button.setEnabled(False)
            progress_bar = self.image_progress
        
        elif file_type == 'video':
            self.compress_video_button.setEnabled(False)
            progress_bar = self.video_progress

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
        output_path, compression_ratio = result

        if file_type == 'image':
            self.compress_image_button.setEnabled(True)

            self.display_image(output_path, self.compressed_image_label)

            size = os.path.getsize(output_path)
            self.compressed_image_size_label.setText(
                f"Size: {self.format_size(size)} (Ratio: {compression_ratio:.2f}x)"
            )

        elif file_type == 'video':
            self.compress_video_button.setEnabled(True)

            self.display_video_frame(output_path, self.compressed_video_label)

            size = os.path.getsize(output_path)
            self.compressed_video_size_label.setText(
                f"Size: {self.format_size(size)} (Ratio: {compression_ratio:.2f}x)"
            )

        QMessageBox.information(
            self,
            "Compression Complete",
            f"File compressed successfully!\nCompression ratio: {compression_ratio:.2f}x"
        )

    def show_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)

        if self.file_type == 'image':
            self.compress_image_button.setEnabled(True)

        elif self.file_type == 'video':
            self.compress_video_button.setEnabled(True)

    def display_image(self, image_path, label):
        try:
            image = Image.open(image_path)

            qimage = ImageQt.ImageQt(image)
            pixmap = QPixmap.fromImage(qimage)

            pixmap = pixmap.scaled(
                label.width(), label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            label.setPixmap(pixmap)

        except Exception as e:
            self.show_error(f"Error displaying image: {str(e)}")

    def display_video_frame(self, video_path, label):
        try:
            cap = cv2.VideoCapture(video_path)

            ret, frame = cap.read()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                height, width, channels = frame_rgb.shape
                bytes_per_line = channels * width
                qimage = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

                pixmap = QPixmap.fromImage(qimage)

                pixmap = pixmap.scaled(
                    label.width(), label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )

                label.setPixmap(pixmap)

            cap.release()

        except Exception as e:
            self.show_error(f"Error displaying video frame: {str(e)}")

    def format_size(self, size_bytes):
        units = ['B', 'KB', 'MB', 'GB', 'TB']

        unit_index = 0
        size = float(size_bytes)

        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1

        if unit_index == 0:
            return f"{int(size)} {units[unit_index]}"
        else:
            return f"{size:.2f} {units[unit_index]}"
