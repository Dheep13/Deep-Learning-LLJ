"""
Age prediction model implementation.
"""

import os
import sys
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import config

def create_data_generators():
    """
    Create data generators for training, validation, and testing.

    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    # Data augmentation for training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Only rescaling for validation and testing
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        os.path.join(config.PROCESSED_DATA_DIR, 'train'),
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        os.path.join(config.PROCESSED_DATA_DIR, 'val'),
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = test_datagen.flow_from_directory(
        os.path.join(config.PROCESSED_DATA_DIR, 'test'),
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator

def build_age_prediction_model(num_classes):
    """
    Build the age prediction model using transfer learning.

    Args:
        num_classes (int): Number of age group classes

    Returns:
        tf.keras.Model: Compiled model
    """
    # Use a pre-trained model as the base
    base_model = applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(*config.IMAGE_SIZE, 3)
    )

    # Freeze the base model layers
    base_model.trainable = False

    # Create the model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_age_prediction_model(model, train_generator, val_generator, epochs=config.NUM_EPOCHS):
    """
    Train the age prediction model.

    Args:
        model (tf.keras.Model): The model to train
        train_generator: Training data generator
        val_generator: Validation data generator
        epochs (int): Number of epochs to train

    Returns:
        tuple: (trained_model, training_history)
    """
    # Create model directory if it doesn't exist
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=os.path.join(config.MODELS_DIR, 'age_prediction_model.keras'),
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
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )

    return model, history

def fine_tune_model(model, train_generator, val_generator, epochs=20):
    """
    Fine-tune the model by unfreezing some layers of the base model.

    Args:
        model (tf.keras.Model): The pre-trained model
        train_generator: Training data generator
        val_generator: Validation data generator
        epochs (int): Number of epochs for fine-tuning

    Returns:
        tuple: (fine-tuned_model, training_history)
    """
    # Unfreeze the last few layers of the base model
    base_model = model.layers[0]
    base_model.trainable = True

    # Freeze all layers except the last 15
    for layer in base_model.layers[:-15]:
        layer.trainable = False

    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=os.path.join(config.MODELS_DIR, 'age_prediction_model_fine_tuned.keras'),
            monitor='val_accuracy',
            save_best_only=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7
        )
    ]

    # Fine-tune the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )

    return model, history

def evaluate_model(model, test_generator):
    """
    Evaluate the model on the test set.

    Args:
        model (tf.keras.Model): The trained model
        test_generator: Test data generator

    Returns:
        tuple: (loss, accuracy)
    """
    return model.evaluate(test_generator)

def predict_age_group(model, image_path):
    """
    Predict the age group for a given image.

    Args:
        model (tf.keras.Model): The trained model
        image_path (str): Path to the image

    Returns:
        tuple: (predicted_age_group, confidence)
    """
    import cv2
    import numpy as np

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

    # Get the class labels from the config
    class_indices = {i: group for i, group in enumerate(config.AGE_GROUPS.keys())}
    predicted_age_group = class_indices[predicted_class_idx]

    return predicted_age_group, confidence
