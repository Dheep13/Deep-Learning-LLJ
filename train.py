"""
Training script for the Facial Age Group Prediction and Identity Verification models.
"""

import os
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import config
from src.models.age_prediction import (
    create_data_generators,
    build_age_prediction_model,
    train_age_prediction_model,
    fine_tune_model,
    evaluate_model
)
from src.models.face_verification import (
    build_embedding_model,
    build_siamese_model,
    create_pairs_dataset,
    train_siamese_model
)
from src.utils.visualization import plot_training_history, plot_confusion_matrix

def train_age_prediction():
    """
    Train the age prediction model.
    """
    print("=== Training Age Prediction Model ===")

    # Create data generators
    print("Creating data generators...")
    train_generator, val_generator, test_generator = create_data_generators()

    # Get number of classes
    num_classes = len(train_generator.class_indices)
    print(f"Number of age group classes: {num_classes}")
    print(f"Class mapping: {train_generator.class_indices}")

    # Build the model
    print("Building age prediction model...")
    model = build_age_prediction_model(num_classes)
    model.summary()

    # Train the model
    print("Training age prediction model...")
    model, history = train_age_prediction_model(model, train_generator, val_generator)

    # Plot training history
    plot_training_history(
        history,
        os.path.join(config.MODELS_DIR, 'age_prediction_training_history.png')
    )

    # Fine-tune the model
    print("Fine-tuning age prediction model...")
    model, ft_history = fine_tune_model(model, train_generator, val_generator)

    # Plot fine-tuning history
    plot_training_history(
        ft_history,
        os.path.join(config.MODELS_DIR, 'age_prediction_fine_tuning_history.png')
    )

    # Evaluate the model
    print("Evaluating age prediction model...")
    loss, accuracy = evaluate_model(model, test_generator)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    # Generate confusion matrix
    print("Generating confusion matrix...")
    y_true = test_generator.classes
    y_pred = np.argmax(model.predict(test_generator), axis=1)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    class_names = list(test_generator.class_indices.keys())
    plot_confusion_matrix(
        cm,
        class_names,
        os.path.join(config.MODELS_DIR, 'age_prediction_confusion_matrix.png')
    )

    # Save the model
    model_path = os.path.join(config.MODELS_DIR, 'age_prediction_model_final.keras')
    model.save(model_path)
    print(f"Model saved to {model_path}")

    return model

def train_face_verification():
    """
    Train the face verification model.
    """
    print("=== Training Face Verification Model ===")

    # Build the embedding model
    print("Building embedding model...")
    embedding_model = build_embedding_model()
    embedding_model.summary()

    # Build the siamese model
    print("Building siamese model...")
    siamese_model, distance_model = build_siamese_model(embedding_model)
    siamese_model.summary()

    # Create pairs dataset
    print("Creating pairs dataset...")
    pairs, labels = create_pairs_dataset(os.path.join(config.PROCESSED_DATA_DIR, 'train'))
    print(f"Created {len(pairs)} pairs ({sum(labels)} genuine, {len(labels) - sum(labels)} impostor)")

    # Train the siamese model
    print("Training siamese model...")
    siamese_model, history = train_siamese_model(siamese_model, pairs, labels)

    # Plot training history
    plot_training_history(
        history,
        os.path.join(config.MODELS_DIR, 'face_verification_training_history.png')
    )

    # Save the models
    siamese_model_path = os.path.join(config.MODELS_DIR, 'face_verification_model_final.keras')
    embedding_model_path = os.path.join(config.MODELS_DIR, 'face_embedding_model.keras')

    siamese_model.save(siamese_model_path)
    embedding_model.save(embedding_model_path)

    print(f"Siamese model saved to {siamese_model_path}")
    print(f"Embedding model saved to {embedding_model_path}")

    return siamese_model

def main():
    """
    Main function to train the models.
    """
    parser = argparse.ArgumentParser(description='Train facial age prediction and identity verification models')
    parser.add_argument('--model', type=str, choices=['age', 'verification', 'both'], default='both',
                        help='Which model to train (age, verification, or both)')
    args = parser.parse_args()

    # Create models directory if it doesn't exist
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # Set memory growth for GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s), memory growth enabled")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")

    # Train the selected models
    if args.model in ['age', 'both']:
        train_age_prediction()

    if args.model in ['verification', 'both']:
        train_face_verification()

    print("Training completed successfully!")

if __name__ == "__main__":
    main()
