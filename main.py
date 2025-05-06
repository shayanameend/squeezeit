#!/usr/bin/env python3
"""
Squeezeit Compression Tool - Main Application

This is the entry point for the Squeezeit Compression Tool, which provides
image and video compression using custom algorithms.
"""

from PyQt6.QtWidgets import QApplication, QMessageBox
import sys
import os

# Rename the new UI file to replace the old one
if os.path.exists('ui/main_window.py.new'):
    if os.path.exists('ui/main_window.py'):
        os.rename('ui/main_window.py', 'ui/main_window.py.bak')
    os.rename('ui/main_window.py.new', 'ui/main_window.py')

from ui.main_window import MainWindow

def main():
    """Main entry point for the application."""
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        # Handle any unexpected exceptions
        error_msg = f"An error occurred: {str(e)}"
        print(error_msg, file=sys.stderr)

        # Try to show an error dialog if possible
        try:
            if QApplication.instance():
                QMessageBox.critical(None, "Error", error_msg)
        except:
            pass

        sys.exit(1)

if __name__ == "__main__":
    main()
