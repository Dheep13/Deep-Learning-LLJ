# Facial Age Group Prediction and Identity Verification

## Project Overview
This project implements deep learning models for facial age group prediction and identity verification using convolutional neural networks and transfer learning.

## Dataset
The dataset consists of facial images categorized into 5 age groups:
- 18-20
- 21-30
- 31-40
- 41-50
- 51-60

## Implementation
The project includes:
1. Data preprocessing and augmentation
2. Model development using transfer learning
3. Training and evaluation
4. Performance analysis

## Model Architecture
- Base model: EfficientNetB0 (pre-trained on ImageNet)
- Custom layers for age group classification
- Training with data augmentation and class weighting

## Results
The model achieves a test accuracy of 20.51% for age group prediction.

## Future Improvements
- Collect more training data
- Implement more sophisticated data augmentation
- Try different model architectures
- Use ensemble methods
