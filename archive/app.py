"""
Simple demo application for the Facial Age Group Prediction and Identity Verification system.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent))

import config
from src.models.face_verification import load_and_preprocess_image
from src.utils.visualization import visualize_face_verification

def predict_age_group(model, image_path):
    """
    Predict the age group for a given image.

    Args:
        model (tf.keras.Model): The trained age prediction model
        image_path (str): Path to the image

    Returns:
        tuple: (predicted_age_group, confidence)
    """
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, config.IMAGE_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Make prediction
    predictions = model.predict(img)

    # Get the predicted class and confidence
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]

    # Get the class labels
    class_indices = {v: k for k, v in enumerate(config.AGE_GROUPS.keys())}
    predicted_age_group = class_indices[predicted_class_idx]

    return predicted_age_group, confidence

def verify_identity(model, image1_path, image2_path):
    """
    Verify if two face images belong to the same person.

    Args:
        model (tf.keras.Model): The trained face verification model
        image1_path (str): Path to the first image
        image2_path (str): Path to the second image

    Returns:
        tuple: (is_same_person, similarity_score)
    """
    # Load and preprocess images
    img1 = load_and_preprocess_image(image1_path)
    img2 = load_and_preprocess_image(image2_path)

    # Add batch dimension
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)

    # Get similarity score
    similarity_score = model.predict([img1, img2])[0][0]

    # Determine if the faces match
    is_same_person = similarity_score >= config.FACE_SIMILARITY_THRESHOLD

    return is_same_person, similarity_score

def detect_and_crop_face(image_path, output_path=None):
    """
    Detect and crop the largest face in an image.

    Args:
        image_path (str): Path to the input image
        output_path (str, optional): Path to save the cropped face

    Returns:
        str: Path to the cropped face image
    """
    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        print(f"No faces detected in {image_path}")
        return None

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
    face_img = cv2.resize(face_img, config.IMAGE_SIZE)

    # Save the cropped face if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, face_img)
        return output_path

    # Otherwise, save to a temporary file
    temp_dir = os.path.join(config.BASE_DIR, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"temp_face_{os.path.basename(image_path)}")
    cv2.imwrite(temp_path, face_img)

    return temp_path

def main():
    """
    Main function for the demo application.
    """
    parser = argparse.ArgumentParser(description='Demo for Facial Age Group Prediction and Identity Verification')
    parser.add_argument('--mode', type=str, choices=['age', 'verification'], required=True,
                        help='Mode: age prediction or face verification')
    parser.add_argument('--image', type=str, help='Path to the image for age prediction')
    parser.add_argument('--image1', type=str, help='Path to the first image for face verification')
    parser.add_argument('--image2', type=str, help='Path to the second image for face verification')
    args = parser.parse_args()

    # Set memory growth for GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")

    # Age prediction mode
    if args.mode == 'age':
        if not args.image:
            parser.error("--image is required for age prediction mode")

        # Load the age prediction model
        model_path = os.path.join(config.MODELS_DIR, 'age_prediction_model_final.keras')
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Please train the model first.")
            return

        model = load_model(model_path)
        print("Age prediction model loaded successfully.")

        # Detect and crop face
        face_path = detect_and_crop_face(args.image)
        if not face_path:
            return

        # Predict age group
        age_group, confidence = predict_age_group(model, face_path)

        # Display results
        print(f"Predicted Age Group: {age_group}")
        print(f"Confidence: {confidence:.4f}")

        # Display the image with prediction
        img = cv2.imread(face_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(f"Predicted Age Group: {age_group} (Confidence: {confidence:.4f})")
        plt.axis('off')
        plt.show()

    # Face verification mode
    elif args.mode == 'verification':
        if not args.image1 or not args.image2:
            parser.error("--image1 and --image2 are required for face verification mode")

        # Load the face verification model
        model_path = os.path.join(config.MODELS_DIR, 'face_verification_model_final.keras')
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Please train the model first.")
            return

        model = load_model(model_path)
        print("Face verification model loaded successfully.")

        # Detect and crop faces
        face1_path = detect_and_crop_face(args.image1)
        face2_path = detect_and_crop_face(args.image2)

        if not face1_path or not face2_path:
            return

        # Verify identity
        is_same_person, similarity_score = verify_identity(model, face1_path, face2_path)

        # Display results
        print(f"Similarity Score: {similarity_score:.4f}")
        print(f"Threshold: {config.FACE_SIMILARITY_THRESHOLD}")
        print(f"Result: {'Same person' if is_same_person else 'Different people'}")

        # Visualize the verification
        output_path = os.path.join(config.BASE_DIR, 'verification_result.png')
        visualize_face_verification(
            face1_path,
            face2_path,
            similarity_score,
            config.FACE_SIMILARITY_THRESHOLD,
            output_path
        )

        # Display the visualization
        img = cv2.imread(output_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()
