"""
Visualization utilities for the Facial Age Group Prediction and Identity Verification project.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import sys

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import config

def plot_sample_images(data_dir, n_samples=3):
    """
    Plot sample images from each age group and split.
    
    Args:
        data_dir (str): Path to the processed data directory
        n_samples (int): Number of samples to plot per age group
    """
    fig, axes = plt.subplots(len(config.AGE_GROUPS), 3, figsize=(15, 4 * len(config.AGE_GROUPS)))
    
    for i, age_group in enumerate(config.AGE_GROUPS.keys()):
        for j, split in enumerate(['train', 'val', 'test']):
            # Get image paths for this age group and split
            image_dir = os.path.join(data_dir, split, age_group)
            if not os.path.exists(image_dir):
                axes[i, j].text(0.5, 0.5, f"No images for {age_group} in {split}", 
                               ha='center', va='center')
                axes[i, j].axis('off')
                continue
                
            image_files = [f for f in os.listdir(image_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                axes[i, j].text(0.5, 0.5, f"No images for {age_group} in {split}", 
                               ha='center', va='center')
                axes[i, j].axis('off')
                continue
            
            # Select a random sample
            sample = random.choice(image_files)
            image_path = os.path.join(image_dir, sample)
            
            # Read and display the image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            axes[i, j].imshow(image)
            axes[i, j].set_title(f"{split.capitalize()}: {age_group}")
            axes[i, j].axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(data_dir, 'sample_images.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Sample images saved to {output_path}")

def plot_training_history(history, output_path):
    """
    Plot training history.
    
    Args:
        history: Training history object from model.fit()
        output_path (str): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Training history plot saved to {output_path}")

def plot_confusion_matrix(cm, class_names, output_path):
    """
    Plot confusion matrix.
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        class_names (list): List of class names
        output_path (str): Path to save the plot
    """
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Confusion matrix plot saved to {output_path}")

def visualize_face_verification(image1_path, image2_path, similarity_score, threshold, output_path):
    """
    Visualize face verification results.
    
    Args:
        image1_path (str): Path to the first image
        image2_path (str): Path to the second image
        similarity_score (float): Similarity score between the two faces
        threshold (float): Threshold for face matching
        output_path (str): Path to save the visualization
    """
    # Read images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    # Convert to RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Display images
    ax1.imshow(img1)
    ax1.set_title('Image 1')
    ax1.axis('off')
    
    ax2.imshow(img2)
    ax2.set_title('Image 2')
    ax2.axis('off')
    
    # Add similarity information
    match_result = "Match" if similarity_score >= threshold else "No Match"
    plt.suptitle(f'Similarity: {similarity_score:.4f} | Threshold: {threshold:.4f} | Result: {match_result}', 
                 fontsize=14, color='green' if match_result == "Match" else 'red')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Face verification visualization saved to {output_path}")
