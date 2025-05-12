"""
Configuration parameters for the Facial Age Group Prediction and Identity Verification project.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Dataset parameters
DATASET_URL = "https://www.kaggle.com/datasets/trainingdatapro/age-detection-human-faces-18-60-years"
KAGGLE_DATASET = "trainingdatapro/age-detection-human-faces-18-60-years"

# Data preprocessing parameters
IMAGE_SIZE = (224, 224)  # Standard input size for many CNN architectures
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Age group bins (in years)
AGE_GROUPS = {
    '18-20': (18, 20),
    '21-30': (21, 30),
    '31-40': (31, 40),
    '41-50': (41, 50),
    '51-60': (51, 60)
}

# Model parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Face verification parameters
FACE_SIMILARITY_THRESHOLD = 0.6  # Threshold for face matching
