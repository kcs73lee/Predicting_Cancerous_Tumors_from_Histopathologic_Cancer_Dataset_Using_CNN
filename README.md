# Predicting Cancerous Tumors from Histopathologic Cancer Dataset Using CNN

## Introduction

This project tackles a binary image classification task focused on identifying **metastatic cancer** in small patches extracted from larger digital pathology scans. The goal is to train a model to distinguish between **cancerous** (`1`) and **non-cancerous** (`0`) tissue samples based on image data.

Each image in the dataset is 32x32 pixels in size. A positive label (`1`) indicates the presence of **at least one pixel** of tumor tissue at the center of the image.

The dataset includes:
- Training images and corresponding labels
- Testing images for predictions
- A sample submission file for evaluation

Full dataset and competition details can be found on [Kaggle](https://www.kaggle.com/competitions/histopathologic-cancer-detection/data).

---

## Project Workflow

### 1. Setting Up the Environment
- Import required libraries including `TensorFlow`, `Keras`, and `sklearn`
- Configure GPU settings for efficient training

### 2. Exploratory Data Analysis (EDA)
- Summary statistics and visualizations of the training labels and image data
- Check for class imbalance and perform cleaning if necessary

### 3. Data Preprocessing
- Normalize pixel values (scale 0–1)
- Perform **data augmentation**:
  - Rotation
  - Zoom
  - Horizontal and vertical flips

### 4. Train/Validation Split
- Use an image data generator to split the dataset:
  - 80% for training
  - 20% for validation

### 5. Base Model Architecture
- Build a simple CNN model using TensorFlow Keras
- Compile the model with appropriate optimizer and loss function
- Train the base model on the training set

### 6. Base Model Evaluation
- Evaluate base model performance on validation set using accuracy and loss metrics

### 7. Hyperparameter Tuning & Model Enhancement
- Increase learning rate from `0.0001` to `0.001`
- Add more convolutional layers and dropout layers
- Add normalization layers to improve training stability
- Extend training epochs and steps for better learning

### 8. Tuned Model Evaluation
- Plot and compare training vs validation performance
- Final evaluation metrics to validate model performance

### 9. Prediction on Test Set
- Generate predictions on the test images
- Format submission according to Kaggle requirements
- Submit to retrieve Kaggle score

---

## Discussion & Summary

- Reflections on model performance and training approach
- Discussion of results and comparison with baseline models
- Suggestions for improvement (e.g., using pre-trained CNNs, ensemble methods, etc.)

---

## Requirements

- Python 3.7+
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib / Seaborn
- Scikit-learn

---

