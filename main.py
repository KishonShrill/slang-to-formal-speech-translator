import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont
from app.ui.translator_window import TranslatorApp

# --- Main execution ---
if __name__ == '__main__':
    # Ensure a QApplication instance exists
    app = QApplication(sys.argv)
    
    # Set a default font
    app.setFont(QFont("Inter", 10))
    
    # Create and show the main window
    window = TranslatorApp()
    window.show()
    
    # Start the event loop
    sys.exit(app.exec())
