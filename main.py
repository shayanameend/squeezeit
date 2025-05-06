import sys
from PyQt6.QtWidgets import QApplication, QMessageBox

from ui.main_window import MainWindow

def main():
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        print(error_msg, file=sys.stderr)

        try:
            if QApplication.instance():
                QMessageBox.critical(None, "Error", error_msg)
        except:
            pass

        sys.exit(1)

if __name__ == "__main__":
    main()
