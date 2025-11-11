from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QPushButton, QComboBox, QLabel, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from app.core.translator_model import translate_text

class TranslatorApp(QWidget):
    """
    GUI for text-based translation model.
    """

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.load_styles()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # --- Title ---
        title_label = QLabel("💬 Gen Z Slang Translator")
        title_label.setObjectName("TitleLabel")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        # --- Direction Selector ---
        direction_layout = QHBoxLayout()
        direction_layout.addWidget(QLabel("Translate:"))

        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["English to Gen Z", "Gen Z to English"])
        self.direction_combo.setFont(QFont("Inter", 10))
        direction_layout.addWidget(self.direction_combo)
        direction_layout.setStretch(1, 1)
        main_layout.addLayout(direction_layout)

        # --- Input ---
        main_layout.addWidget(QLabel("Input Text:"))
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Type or paste your text here...")
        self.input_text.setFont(QFont("Inter", 11))
        self.input_text.setMinimumHeight(150)
        main_layout.addWidget(self.input_text)

        # --- Button ---
        self.translate_button = QPushButton("Translate")
        self.translate_button.setObjectName("TranslateButton")
        self.translate_button.setMinimumHeight(45)
        self.translate_button.clicked.connect(self.on_translate)
        main_layout.addWidget(self.translate_button)

        # --- Separator ---
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(separator)

        # --- Output ---
        main_layout.addWidget(QLabel("Output:"))
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setPlaceholderText("Translation will appear here...")
        self.output_text.setFont(QFont("Inter", 11))
        self.output_text.setMinimumHeight(150)
        main_layout.addWidget(self.output_text)

        # --- Set Layout ---
        self.setLayout(main_layout)
        self.setWindowTitle("Translator GUI")
        self.setGeometry(300, 300, 500, 600)

    def load_styles(self):
        """Load QSS stylesheet."""
        with open("app/styles/styles.qss", "r") as f:
            self.setStyleSheet(f.read())

    def on_translate(self):
        """Handle Translate button click."""
        input_str = self.input_text.toPlainText().strip()
        direction_text = self.direction_combo.currentText()

        if not input_str:
            self.output_text.setText("Please enter some text to translate.")
            return

        direction = "eng_to_genz" if direction_text == "English to Gen Z" else "genz_to_eng"

        try:
            result = translate_text(input_str, direction)
            self.output_text.setText(result)
        except Exception as e:
            self.output_text.setText(f"Error: {e}")

