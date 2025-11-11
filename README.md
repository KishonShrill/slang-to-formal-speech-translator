# Slang to Formal Speech Translator
🧠 Transformer Model Midterm Project
CSC172 & CSC173 — Introduction to Artificial Intelligence

Mindanao State University - Iligan Institute of Technology (MSU-IIT)

## 👥 Group Members

- Member 1: Reambonanza, Kyla C.
- Member 2: Belvis, Febe Gwyn R.
- Member 3: Pingol, Chriscent Louis June M.

## 📝 Project Title

**TransLIT 🔥: English Slang-to-Normal Speech Translator**

## 🎯 Project Overview

This project demonstrates the application of a Transformer model in translating between standard English and Gen Z slang.
It explores how modern NLP (Natural Language Processing) architectures, specifically Transformers, can learn informal linguistic patterns and contextually map slang expressions to their standard equivalents.

## ⚙️ Objectives

1. Implement and experiment with a Transformer architecture for language translation.
2. Build a user-friendly PyQt6 interface for real-time text translation.
3. Evaluate the model’s accuracy and creativity in translating slang and idiomatic expressions.
4. Present a clear, technically sound, and creative application of Transformers.

## 🧩 System Architecture
**Components**

1. Frontend (UI):
- Built using PyQt6 for desktop GUI interaction.
- Allows users to select translation direction and view output.

2. Model Backend:
- Initially uses a rule-based placeholder (dictionary).
- To be upgraded with a Transformer-based model, such as:
    - MarianMT (for text translation) [NOT FINAL]
    - or DistilGPT2 fine-tuned on Gen Z slang datasets. [NOT FINAL]

3. Integration Layer:
- Connects the PyQt6 interface with the model backend through modular architecture (app/core/translator_model.py).

## 🧱 Folder Structure
```bash
genz_translator/
│
├── main.py                      # Entry point
│
├── app/
│   ├── ui/
│   │   └── translator_window.py # Main PyQt6 GUI
│   ├── core/
│   │   └── translator_model.py  # Model logic (Transformer integration)
│   └── styles/
│       └── styles.qss           # Stylesheet
│
├── assets/
│   └── icons/
│       └── down_arrow.png
│
└── requirements.txt
```

## 🧪 Model Details

- Architecture: Transformer (Encoder-Decoder or Decoder-only)
- Frameworks: PyTorch, Hugging Face Transformers
- Dataset (planned):
    - Custom dataset mapping English ↔ Gen Z slang
    - Optionally fine-tuned from existing translation models.

## 🚀 How to Run
```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows

# or
echo "PIPENV_VENV_IN_PROJECT=1" >> .env
pipenv shell

# 2. Install dependencies
pip install -r requirements.txt

# or
pipenv install --dev

# 3. Run the application
python main.py
```

## 📊 Evaluation Criteria (from Midterm Instructions)

Each group will be evaluated based on:

- Clarity
- Technical soundness
- Creativity
- Presentation quality

## 🧾 Deliverables

- Working Transformer-based translator prototype
- GitHub repository with code and README
- Group presentation (10 minutes)
- Peer feedback for at least one other group

## 💡 Future Improvements [NOT FINAL]

- Integrate pretrained Transformer-based translation models
- Add dataset collection and fine-tuning pipeline
- Evaluate BLEU/ROUGE scores for translation quality
- Expand UI with text-to-speech and conversation features
