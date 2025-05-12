"""
Face verification model implementation using Siamese Networks.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from pathlib import Path

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import config

def build_embedding_model():
    """
    Build a model to extract face embeddings.

    Returns:
        tf.keras.Model: Model that extracts face embeddings
    """
    # Use a pre-trained model as the base
    base_model = applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(*config.IMAGE_SIZE, 3)
    )

    # Freeze the base model layers
    base_model.trainable = False

    # Create the embedding model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(256, activation=None),  # Embedding layer
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))  # L2 normalization
    ])

    return model

def build_siamese_model(embedding_model):
    """
    Build a Siamese network for face verification.

    Args:
        embedding_model (tf.keras.Model): Model that extracts face embeddings

    Returns:
        tuple: (siamese_model, distance_model)
    """
    # Input layers for the two images
    input_image1 = layers.Input(shape=(*config.IMAGE_SIZE, 3))
    input_image2 = layers.Input(shape=(*config.IMAGE_SIZE, 3))

    # Get embeddings for both images
    embedding1 = embedding_model(input_image1)
    embedding2 = embedding_model(input_image2)

    # Calculate L1 distance between embeddings
    l1_distance = layers.Lambda(
        lambda embeddings: tf.abs(embeddings[0] - embeddings[1])
    )([embedding1, embedding2])

    # Add dense layers to predict if the faces match
    x = layers.Dense(128, activation='relu')(l1_distance)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    # Create the siamese model
    siamese_model = models.Model(
        inputs=[input_image1, input_image2],
        outputs=outputs
    )

    # Compile the model
    siamese_model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Create a model to compute the distance between two face embeddings
    distance_model = models.Model(
        inputs=[input_image1, input_image2],
        outputs=l1_distance
    )

    return siamese_model, distance_model

def create_pairs_dataset(data_dir, num_pairs=1000):
    """
    Create pairs of images for training the Siamese network.

    Args:
        data_dir (str): Path to the processed data directory
        num_pairs (int): Number of pairs to create

    Returns:
        tuple: (pairs, labels)
    """
    import random
    import os
    import cv2

    # Get all image paths
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    # Create pairs
    pairs = []
    labels = []

    for _ in range(num_pairs):
        # Decide if this will be a genuine pair (same person) or impostor pair (different people)
        is_genuine = random.random() > 0.5

        if is_genuine:
            # Select a random image
            img_path = random.choice(image_paths)

            # Get all images of the same person
            person_id = os.path.basename(img_path).split('_')[0]  # Assuming filename format: person_id_*.jpg
            same_person_images = [p for p in image_paths if os.path.basename(p).split('_')[0] == person_id]

            # If there's only one image of this person, create an impostor pair instead
            if len(same_person_images) < 2:
                is_genuine = False
            else:
                # Select two different images of the same person
                img1_path = img_path
                img2_path = random.choice([p for p in same_person_images if p != img1_path])

                pairs.append((img1_path, img2_path))
                labels.append(1)  # 1 for genuine pair

        if not is_genuine:
            # Select two random images of different people
            img1_path = random.choice(image_paths)
            person1_id = os.path.basename(img1_path).split('_')[0]

            # Get images of different people
            different_person_images = [p for p in image_paths if os.path.basename(p).split('_')[0] != person1_id]

            # If there are no images of different people, skip this pair
            if not different_person_images:
                continue

            img2_path = random.choice(different_person_images)

            pairs.append((img1_path, img2_path))
            labels.append(0)  # 0 for impostor pair

    return pairs, labels

def load_and_preprocess_image(image_path):
    """
    Load and preprocess an image.

    Args:
        image_path (str): Path to the image

    Returns:
        numpy.ndarray: Preprocessed image
    """
    import cv2

    # Read the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to target size
    img = cv2.resize(img, config.IMAGE_SIZE)

    # Normalize pixel values
    img = img / 255.0

    return img

def train_siamese_model(siamese_model, pairs, labels, val_split=0.2, epochs=config.NUM_EPOCHS):
    """
    Train the Siamese network.

    Args:
        siamese_model (tf.keras.Model): The Siamese model
        pairs (list): List of image pairs
        labels (list): List of labels (1 for genuine, 0 for impostor)
        val_split (float): Validation split ratio
        epochs (int): Number of epochs to train

    Returns:
        tuple: (trained_model, training_history)
    """
    import numpy as np

    # Create model directory if it doesn't exist
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # Load and preprocess all images
    print("Loading and preprocessing images...")
    images1 = []
    images2 = []

    for img1_path, img2_path in pairs:
        img1 = load_and_preprocess_image(img1_path)
        img2 = load_and_preprocess_image(img2_path)

        images1.append(img1)
        images2.append(img2)

    images1 = np.array(images1)
    images2 = np.array(images2)
    labels = np.array(labels)

    # Split into training and validation sets
    indices = np.arange(len(labels))
    np.random.shuffle(indices)

    val_size = int(len(indices) * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_images1 = images1[train_indices]
    train_images2 = images2[train_indices]
    train_labels = labels[train_indices]

    val_images1 = images1[val_indices]
    val_images2 = images2[val_indices]
    val_labels = labels[val_indices]

    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=os.path.join(config.MODELS_DIR, 'face_verification_model.keras'),
            monitor='val_accuracy',
            save_best_only=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
    ]

    # Train the model
    history = siamese_model.fit(
        [train_images1, train_images2],
        train_labels,
        validation_data=([val_images1, val_images2], val_labels),
        batch_size=config.BATCH_SIZE,
        epochs=epochs,
        callbacks=callbacks
    )

    return siamese_model, history

def verify_faces(model, image1_path, image2_path, threshold=config.FACE_SIMILARITY_THRESHOLD):
    """
    Verify if two face images belong to the same person.

    Args:
        model (tf.keras.Model): The trained Siamese model
        image1_path (str): Path to the first image
        image2_path (str): Path to the second image
        threshold (float): Threshold for face matching

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
    is_same_person = similarity_score >= threshold

    return is_same_person, similarity_score
