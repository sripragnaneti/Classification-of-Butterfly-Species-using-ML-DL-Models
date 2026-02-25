# Butterfly Species Classification using Machine Learning and Deep Learning

## Overview

This project focuses on classifying butterfly species using both classical Machine Learning algorithms and a Deep Learning architecture.

Two different approaches are implemented and compared:

- Random Forest (Classical ML)
- MobileNetV2 (Transfer Learning - Deep Learning)

The goal is to analyze performance differences between feature-based machine learning models and end-to-end deep learning models.

---

## Dataset

This project uses the Butterfly Image Classification dataset available on Kaggle:

https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification

Dataset Details:
- 75 butterfly species
- 6490 labeled images
- 5192 training images
- 1298 testing images

Note:
The original dataset images are not included in this repository due to licensing restrictions.  
Please download the dataset directly from Kaggle.

---

## Feature-Based Approach (Classical ML)

The classical ML model is trained on extracted visual features including:

- Dominant Colors (K-Means clustering)
- HSV Color Histogram
- GLCM texture features
- LBP Histogram
- Hu Moments
- HOG descriptors

Total features per image: 212

Model implemented:
- Random Forest Classifier

---

## Deep Learning Approach

The deep learning model uses:

- MobileNetV2 (pretrained on ImageNet)
- Transfer Learning
- Fine-tuning of selected layers
- Early Stopping to prevent overfitting

Images are resized to 224x224 and normalized before training.

---

## Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas
- OpenCV
- Matplotlib

