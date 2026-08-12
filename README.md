# cancer-detection-ml
Binary CNN classifier for cancer detection using the Wisconsin dataset — ~90%+ test accuracy
# Cancer Detection using Machine Learning

A binary classification model to detect cancerous vs. non-cancerous samples using
Convolutional Neural Networks (CNN), trained and evaluated on the Wisconsin dataset.

## Overview
This project applies deep learning to a medical diagnosis task, comparing CNN-based
classification against traditional ML baselines (SVM, Logistic Regression, and KNN).

## Tech Stack
- Python
- TensorFlow / Keras
- scikit-learn
- CNN (Convolutional Neural Network)

## Approach
- Built a binary CNN classification model to detect cancerous vs. non-cancerous samples
- Applied image augmentation and dropout regularization to reduce overfitting
- Compared CNN performance against baseline SVM and Logistic Regression models
- Achieved ~90%+ test accuracy

## Results
- Test accuracy: ~90%+
- Outperformed baseline SVM and Logistic Regression models on the same dataset

## How to Run
1. Clone the repo
2. Install dependencies: `pip install tensorflow scikit-learn numpy pandas`
3. Run the training script: `python train.py`

## Future Work
- Hyperparameter tuning for further accuracy gains
- Cross-validation across multiple dataset splits
