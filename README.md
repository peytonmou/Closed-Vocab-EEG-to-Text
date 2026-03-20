# EEG-to-Text Decoding: Imagined Speech & Silent Reading

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![MNE](https://img.shields.io/badge/MNE-1.5+-green.svg)](https://mne.tools/)

## Overview
This project develops deep learning models for decoding imagined speech and silent reading from 64-channel EEG signals. We collected and processed neural signals from 12 participants to evaluate multiple architectures under different validation schemes.

**Key Achievement**: EEGNet achieved **27.05% test accuracy** on 10-class imagined speech decoding 

## Features
- **64-channel EEG dataset** from 12 participants
- **Two paradigms**: Imagined speech (audio) and silent reading
- **Three model architectures**: EEGNet, NiceEEG, NetTraST
- **Three evaluation protocols**: Within-subject, 5-fold Cross-Validation, LOSO (Leave one subject out)
- **Comprehensive preprocessing** with MNE-Python
- **Automated experiment tracking** with metrics and confusion matrices

## Dataset
- **Participants**: 12 healthy subjects
- **Channels**: 64 EEG electrodes
- **Tasks**: 
  - Imagined speech (audio prompts)
  - Silent reading (text presentation)
- **Classes**: 10 words

## Models Implemented

| Model | Description | Key Features |
|-------|-------------|--------------|
| **EEGNet** | Compact CNN for EEG | Depthwise/Separable conv |
| **NiceEEG** | Contrastive learning | Projection head, instance discrimination |
| **NetTraST** | Transformer-based | Multi-head attention, positional encoding |

## Evaluation Protocols
1. **Within-Subject**: Train/test on same subject
2. **5-Fold Cross-Validation**: Subject-independent folds
3. **LOSO**: Leave-One-Subject-Out for generalization

## Installation

```bash
git clone https://github.com/yourusername/eeg-text-decoding.git
cd eeg-text-decoding

# Install dependencies
pip install torch torchvision torchaudio
pip install mne numpy pandas scikit-learn
pip install pyyaml tqdm matplotlib

# For model-specific requirements
pip install einops tensorboard
