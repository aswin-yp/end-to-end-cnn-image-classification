# End-to-End Image Classification using CNN

## Problem Statement
Build an end-to-end deep learning system to classify food images into three categories: Pizza, Steak, and Sushi, and deploy the trained model for real-time inference.

## Dataset
- Image dataset containing food images categorized into Pizza, Steak, and Sushi
- Train/Test split organized using folder structure
- Source: Public image dataset (used for learning and experimentation)

## Approach
- Performed data inspection and preprocessing (resizing, normalization)
- Applied data augmentation to improve model generalization
- Designed and trained a Convolutional Neural Network (CNN) using PyTorch
- Used custom Dataset and DataLoader pipelines
- Evaluated model using training/testing accuracy and loss curves
- Compared baseline vs augmented models
- Deployed the trained model as a REST API using FastAPI

## Tech Stack
- Python
- PyTorch, torchvision
- FastAPI
- NumPy, Matplotlib

## Results
- Achieved stable classification performance across all three classes
- Data augmentation improved generalization on unseen images
- Successfully served predictions via REST API

## How to Run
1. Install dependencies:
2. Train the model:
3. Run API:

## Future Improvements
- Add model versioning and monitoring
- Experiment with transfer learning (ResNet, EfficientNet)
- Deploy using Docker
