"""
Evaluation script for the Facial Age Group Prediction and Identity Verification models.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

import config
from src.models.age_prediction import create_data_generators
from src.utils.visualization import plot_confusion_matrix, visualize_face_verification
from src.models.face_verification import load_and_preprocess_image

def evaluate_age_prediction_model():
    """
    Evaluate the age prediction model on the test set.
    """
    print("=== Evaluating Age Prediction Model ===")

    # Load the model
    model_path = os.path.join(config.MODELS_DIR, 'age_prediction_model_final.keras')
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return

    model = load_model(model_path)
    print("Model loaded successfully.")

    # Create data generators
    _, _, test_generator = create_data_generators()

    # Evaluate the model
    print("Evaluating model on test set...")
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    # Get predictions
    y_true = test_generator.classes
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Generate classification report
    class_names = list(test_generator.class_indices.keys())

    # Check which classes are present in the test set
    unique_classes = np.unique(y_true)
    present_class_names = [class_names[i] for i in unique_classes]

    # Check which classes are predicted
    unique_preds = np.unique(y_pred)

    print(f"\nClasses in test set: {present_class_names}")
    print(f"Classes predicted: {[class_names[i] for i in unique_preds]}")

    try:
        # Get the classes that are actually present in both y_true and y_pred
        present_classes = sorted(set(y_true) | set(y_pred))
        present_class_names = [class_names[i] for i in present_classes]

        # Generate report only for present classes
        report = classification_report(
            y_true, y_pred,
            labels=present_classes,
            target_names=present_class_names,
            zero_division=0
        )
        print("\nClassification Report:")
        print(report)
    except Exception as e:
        print(f"Error generating classification report: {e}")
        report = f"Error generating report: {e}"

    # Generate confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        plot_confusion_matrix(
            cm,
            class_names,
            os.path.join(config.MODELS_DIR, 'age_prediction_confusion_matrix_eval.png')
        )
        print(f"Confusion matrix saved to {os.path.join(config.MODELS_DIR, 'age_prediction_confusion_matrix_eval.png')}")
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")

    # Save classification report to file
    with open(os.path.join(config.MODELS_DIR, 'age_prediction_classification_report.txt'), 'w') as f:
        f.write("Age Prediction Model Evaluation\n")
        f.write("==============================\n\n")
        f.write(f"Test loss: {loss:.4f}\n")
        f.write(f"Test accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"Evaluation results saved to {config.MODELS_DIR}")

def evaluate_face_verification_model():
    """
    Evaluate the face verification model on a test set of face pairs.
    """
    print("=== Evaluating Face Verification Model ===")

    # Load the model
    model_path = os.path.join(config.MODELS_DIR, 'face_verification_model_final.keras')
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return

    model = load_model(model_path)
    print("Model loaded successfully.")

    # Create test pairs
    from src.models.face_verification import create_pairs_dataset

    print("Creating test pairs...")
    test_pairs, test_labels = create_pairs_dataset(
        os.path.join(config.PROCESSED_DATA_DIR, 'test'),
        num_pairs=500
    )
    print(f"Created {len(test_pairs)} test pairs ({sum(test_labels)} genuine, {len(test_labels) - sum(test_labels)} impostor)")

    # Evaluate on test pairs
    print("Evaluating model on test pairs...")

    # Load and preprocess test images
    test_images1 = []
    test_images2 = []

    for img1_path, img2_path in test_pairs:
        img1 = load_and_preprocess_image(img1_path)
        img2 = load_and_preprocess_image(img2_path)

        test_images1.append(img1)
        test_images2.append(img2)

    test_images1 = np.array(test_images1)
    test_images2 = np.array(test_images2)
    test_labels = np.array(test_labels)

    # Get predictions
    y_pred_probs = model.predict([test_images1, test_images2])
    y_pred = (y_pred_probs >= config.FACE_SIMILARITY_THRESHOLD).astype(int)

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred)
    recall = recall_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Generate ROC curve
    fpr, tpr, thresholds = roc_curve(test_labels, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # Save ROC curve
    roc_path = os.path.join(config.MODELS_DIR, 'face_verification_roc_curve.png')
    plt.savefig(roc_path)
    plt.close()

    print(f"ROC curve saved to {roc_path}")

    # Visualize some examples
    os.makedirs(os.path.join(config.MODELS_DIR, 'verification_examples'), exist_ok=True)

    # Select random examples
    np.random.seed(config.RANDOM_SEED)
    indices = np.random.choice(len(test_pairs), min(10, len(test_pairs)), replace=False)

    for i, idx in enumerate(indices):
        img1_path, img2_path = test_pairs[idx]
        true_label = test_labels[idx]
        pred_score = y_pred_probs[idx][0]
        pred_label = y_pred[idx][0]

        output_path = os.path.join(
            config.MODELS_DIR,
            'verification_examples',
            f'example_{i+1}_true_{true_label}_pred_{pred_label}.png'
        )

        visualize_face_verification(
            img1_path,
            img2_path,
            pred_score,
            config.FACE_SIMILARITY_THRESHOLD,
            output_path
        )

    # Save evaluation results to file
    with open(os.path.join(config.MODELS_DIR, 'face_verification_evaluation_report.txt'), 'w') as f:
        f.write("Face Verification Model Evaluation\n")
        f.write("=================================\n\n")
        f.write(f"Threshold: {config.FACE_SIMILARITY_THRESHOLD}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")

    print(f"Evaluation results saved to {config.MODELS_DIR}")

def main():
    """
    Main function to evaluate the models.
    """
    parser = argparse.ArgumentParser(description='Evaluate facial age prediction and identity verification models')
    parser.add_argument('--model', type=str, choices=['age', 'verification', 'both'], default='both',
                        help='Which model to evaluate (age, verification, or both)')
    args = parser.parse_args()

    # Set memory growth for GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s), memory growth enabled")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")

    # Evaluate the selected models
    if args.model in ['age', 'both']:
        evaluate_age_prediction_model()

    if args.model in ['verification', 'both']:
        evaluate_face_verification_model()

    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
