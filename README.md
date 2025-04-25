A simple TDNN network that works on syllable level mfcc and classifies disfluencies # TDNN-
# TDNN-based Syllable MFCC Classifier

This project implements a Time Delay Neural Network (TDNN) to classify syllables using Mel-Frequency Cepstral Coefficients (MFCC) extracted from speech audio. The goal is to build a lightweight, syllable-level speech processing model using Python, TensorFlow, and librosa.

## Features

- Extracts MFCC features using `librosa`
- Uses TDNN layers to learn temporal patterns in speech
- Classifies input audio into predefined syllable classes
- Configurable, simple architecture for experimentation

## Project Structure

```
tdnn_syllable_classifier/
├── data/                  # Audio samples (.wav) and their labels
├── scripts/
│   ├── extract_mfcc.py    # Extract MFCCs from audio
│   └── train_tdnn.py      # TDNN training pipeline
├── model/
│   └── tdnn.py            # TDNN model definition
├── utils/
│   └── data_loader.py     # Dataset loader
├── config.yaml            # Model and training configuration
├── README.md
├── requirements.txt
└── .gitignore