# Hand Gesture Recognition using Pretrained CNNs

## Overview
This project utilizes **AlexNet** and **VGG16** pretrained convolutional neural networks for hand gesture recognition. The system extracts features from images using these networks and classifies gestures through a custom **fully connected classifier**. The project evaluates performance on different datasets, including a revised dataset and a custom dataset.

## Features
- **Pretrained CNN Feature Extraction**: Uses AlexNet and VGG16 to extract image features.
- **Custom Fully Connected Classifier**: Implements a multi-layer perceptron to classify hand gestures.
- **Dataset Handling**: Works with predefined and custom datasets.
- **Performance Evaluation**: Includes accuracy measurements, confusion matrices, and hyperparameter tuning.

## Project Structure
```
📂 Hand-Gesture-Recognition
│── 📂 Data                # Contains the dataset (Revised & Unlabeled)
│── 📂 Models              # Trained models and checkpoints
│── 📂 Notebooks           # Jupyter Notebooks for training and evaluation
│── 📜 A2.ipynb            # Main notebook for execution
│── 📜 A2.html             # Exported HTML version
│── 📜 Submission.zip      # Packaged submission files
│── 📜 README.md           # Project documentation
```

## How to Run
1. Open **A2.ipynb** in **Google Colab**.
2. Ensure the dataset is uploaded in the correct structure.
3. Run the notebook cells in sequence to train and evaluate the model.

## Results
- **AlexNet with Revised Data**: **92.38% accuracy**
- **CNN Model with Own Data**: **70.37% accuracy**
- **VGG16 Feature Extraction**: Experimented for improved performance.
- **Confusion Matrices**: Included for model comparison.

## Future Improvements
- **Data Augmentation**: Increase dataset diversity to reduce overfitting.
- **Regularization Techniques**: Experiment with dropout, weight decay, and batch normalization.
- **Advanced Architectures**: Try **ResNet** or **EfficientNet** for potentially better performance.

---

© 2024 Woody Chang. All Rights Reserved.
