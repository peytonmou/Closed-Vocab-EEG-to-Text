EEG-to-Text Decoding

This repository provides implementations for EEG-to-Text decoding experiments using three deep learning models (EEGNet, NetTraST, and NiceEEG) under two paradigms (Silent Reading and Imagined Speech) and three evaluation settings (Within-Subject, Cross-Validation, and Leave-One-Subject-Out (LOSO)).

📁 Repository Structure
.
├── data.py        # Data loading and preprocessing
├── models.py      # EEGNet, NetTraST, NiceEEG implementations
├── metrics.py     # Evaluation metrics
├── main.py        # Main training and evaluation script
└── README.md

Step 1: Download the dataset from Hugging Face:
https://huggingface.co/datasets/peytonmou/EEG2Text

Using Git:
git lfs install
git clone https://huggingface.co/datasets/peytonmou/EEG2Text

Step 2: Clone This Repository:
git clone https://github.com/peytonmou/EEG-to-Text.git
cd EEG-to-Text

Place the dataset folder in the same directory as this repository:
├── EEG-to-Text
│   ├── main.py
│   ├── data.py
│   ├── models.py
│   └── metrics.py
└── EEG2Text   ← dataset folder

Step 3: Run the main script:
python main.py

Inside main.py, you can configure the experiment by selecting:
Data Types: Silent Reading; Imagined Speech
Models: EEGNet; NetTraST; NiceEEG
Evaluation Settings: Within-Subject; Cross-Validation; LOSO (Leave-One-Subject-Out)

Requirements:
Python ≥ 3.8
PyTorch
NumPy
SciPy
scikit-learn

Install dependencies:
pip install torch numpy scipy scikit-learn
Silent Reading is more challenging due to weaker semantic-related neural signals.

The code supports reproducible evaluation under three experimental protocols.
