"""
Script for preprocessing the facial images dataset.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil
import random
from sklearn.model_selection import train_test_split

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import config
from src.utils.visualization import plot_sample_images

def detect_faces(image_path):
    """
    Detect faces in an image and return the face regions.

    Args:
        image_path (str): Path to the image file

    Returns:
        list: List of detected face regions (x, y, w, h)
    """
    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return []

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    return faces

def preprocess_image(image_path, output_path, target_size=config.IMAGE_SIZE):
    """
    Preprocess an image:
    1. Detect faces
    2. Crop the largest face
    3. Resize to target size
    4. Normalize pixel values

    Args:
        image_path (str): Path to the input image
        output_path (str): Path to save the preprocessed image
        target_size (tuple): Target image size (width, height)

    Returns:
        bool: True if preprocessing was successful, False otherwise
    """
    try:
        # Detect faces
        faces = detect_faces(image_path)

        if len(faces) == 0:
            print(f"No faces detected in {image_path}")
            return False

        # Read the original image
        image = cv2.imread(image_path)

        # Find the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face

        # Add some margin (20%)
        margin = int(0.2 * max(w, h))
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)

        # Crop the face
        face_img = image[y:y+h, x:x+w]

        # Resize to target size
        face_img = cv2.resize(face_img, target_size)

        # Save the preprocessed image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, face_img)

        return True

    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        return False

def create_dataset_structure():
    """
    Create the dataset structure for training, validation, and testing.

    The structure will be:
    processed/
    ├── train/
    │   ├── 18-25/
    │   ├── 26-35/
    │   ├── 36-45/
    │   └── 46-60/
    ├── val/
    │   ├── 18-25/
    │   ├── 26-35/
    │   ├── 36-45/
    │   └── 46-60/
    └── test/
        ├── 18-25/
        ├── 26-35/
        ├── 36-45/
        └── 46-60/
    """
    # Create directories
    for split in ['train', 'val', 'test']:
        for age_group in config.AGE_GROUPS.keys():
            os.makedirs(os.path.join(config.PROCESSED_DATA_DIR, split, age_group), exist_ok=True)

def get_age_group(age):
    """
    Determine the age group for a given age.

    Args:
        age (int): Age in years

    Returns:
        str: Age group label
    """
    for group, (min_age, max_age) in config.AGE_GROUPS.items():
        if min_age <= age <= max_age:
            return group
    return None

def preprocess_dataset():
    """
    Preprocess the entire dataset.
    """
    print("Starting dataset preprocessing...")

    # Create dataset structure
    create_dataset_structure()

    # Define paths to the existing dataset structure
    train_dir = os.path.join(config.DATA_DIR, 'train')
    test_dir = os.path.join(config.DATA_DIR, 'test')

    # Get list of all image files and their age groups
    train_files = []
    train_labels = []
    test_files = []
    test_labels = []

    # Process training images
    print("Finding training images...")
    for age_group in os.listdir(train_dir):
        age_group_dir = os.path.join(train_dir, age_group)
        if os.path.isdir(age_group_dir):
            for file in os.listdir(age_group_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(age_group_dir, file)
                    train_files.append(image_path)
                    train_labels.append(age_group)

    # Process test images
    print("Finding test images...")
    for age_group in os.listdir(test_dir):
        age_group_dir = os.path.join(test_dir, age_group)
        if os.path.isdir(age_group_dir):
            for file in os.listdir(age_group_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(age_group_dir, file)
                    test_files.append(image_path)
                    test_labels.append(age_group)

    print(f"Found {len(train_files)} training images and {len(test_files)} test images")

    # Split the training set to create a validation set
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files, train_labels,
        test_size=config.VAL_RATIO,
        stratify=train_labels,
        random_state=config.RANDOM_SEED
    )

    # Process each split
    splits = [
        ('train', train_files, train_labels),
        ('val', val_files, val_labels),
        ('test', test_files, test_labels)
    ]

    for split_name, files, labels in splits:
        print(f"Processing {split_name} split ({len(files)} images)...")

        for i, (image_path, age_group) in enumerate(tqdm(zip(files, labels), total=len(files))):
            output_path = os.path.join(
                config.PROCESSED_DATA_DIR,
                split_name,
                age_group,
                f"{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
            )

            preprocess_image(image_path, output_path)

    print("Dataset preprocessing completed!")

    # Generate and save dataset statistics
    generate_dataset_stats()

    # Plot sample images
    plot_sample_images(config.PROCESSED_DATA_DIR)

def generate_dataset_stats():
    """
    Generate and save dataset statistics.
    """
    stats = {'split': [], 'age_group': [], 'count': []}

    for split in ['train', 'val', 'test']:
        for age_group in config.AGE_GROUPS.keys():
            path = os.path.join(config.PROCESSED_DATA_DIR, split, age_group)
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                stats['split'].append(split)
                stats['age_group'].append(age_group)
                stats['count'].append(count)

    # Create DataFrame
    df = pd.DataFrame(stats)

    # Save statistics
    stats_path = os.path.join(config.PROCESSED_DATA_DIR, 'dataset_stats.csv')
    df.to_csv(stats_path, index=False)

    print(f"Dataset statistics saved to {stats_path}")
    print(df.pivot(index='age_group', columns='split', values='count'))

if __name__ == "__main__":
    preprocess_dataset()
