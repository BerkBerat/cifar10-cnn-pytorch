# 🧠 CIFAR-10 Image Classification with PyTorch CNN

This repository contains a **PyTorch implementation** of a **Convolutional Neural Network (CNN)** designed for image classification on the **CIFAR-10 dataset**. The project demonstrates the full lifecycle of building, training, and evaluating a CNN for a multi-class image classification problem.

---

## 📚 Project Overview

The **CIFAR-10** dataset consists of **60,000 32x32 color images** in **10 classes**, with **6,000 images per class**.  
The goal of this project is to classify these images into the following categories:

- ✈️ airplane
- 🚗 automobile
- 🐦 bird
- 🐱 cat
- 🦌 deer
- 🐶 dog
- 🐸 frog
- 🐴 horse
- 🚢 ship
- 🚚 truck

---

## ✨ Features

✅ Efficient data loading and preprocessing using `torchvision`  
✅ Custom CNN model with convolutional, pooling, activation, dropout and fully connected layers  
✅ Utility functions for visualizing sample images  
✅ Training loop with loss tracking  
✅ Evaluation of model accuracy on both train and test sets  
✅ Real-time plotting of training loss

---

## 🛠️ Requirements

Make sure you have Python 3.x and the following packages installed:

- `torch`
- `torchvision`
- `matplotlib`
- `numpy`

Certainly! This "Model Architecture" section can be made much more readable and visually appealing in a README. Here are a few improved versions, using different formatting styles.

Option 1: Clearer Step-by-Step with Headings (Recommended)

This approach breaks down the architecture into logical blocks, making it very easy to follow the data flow.

Markdown

## Model Architecture

The Convolutional Neural Network (CNN) is designed to process 32x32 color images (input shape: 3x32x32) through a series of convolutional, pooling, and fully connected layers.

### 1. Feature Extraction (Convolutional Blocks)

The initial layers are responsible for extracting features from the input image.

* **Input Layer**: 3x32x32 color image
* **First Convolutional Block**:
    * `Conv2d(3, 32, kernel_size=3, padding=1)`: Applies 32 filters of size 3x3 with padding, converting the 3-channel input to 32 output channels.
    * `ReLU`: Applies the Rectified Linear Unit activation function.
    * `MaxPool2d(kernel_size=2, stride=2)`: Reduces the spatial dimensions by a factor of 2 (e.g., from 32x32 to 16x16).
* **Second Convolutional Block**:
    * `Conv2d(32, 64, kernel_size=3, padding=1)`: Applies 64 filters to the 32-channel input, resulting in 64 output channels.
    * `ReLU`: Applies the Rectified Linear Unit activation function.
    * `MaxPool2d(kernel_size=2, stride=2)`: Further reduces the spatial dimensions (e.g., from 16x16 to 8x8).

### 2. Classification (Fully Connected Layers)

After feature extraction, the flattened features are fed into fully connected layers for classification.

* **Flatten Layer**: Reshapes the output from the convolutional layers (e.g., 64 channels * 8x8 spatial dimensions = 4096 features) into a 1D vector.
* **First Fully Connected Layer**:
    * `Linear(64*8*8, 128)`: Connects the flattened features to 128 output neurons.
    * `ReLU`: Applies the Rectified Linear Unit activation function.
    * `Dropout(p=0.2)`: Randomly zeroes some of the elements with probability `p` during training, helping to prevent overfitting.
* **Output Layer**:
    * `Linear(128, 10)`: Maps the 128 features to 10 output classes (corresponding to the 10 classes in CIFAR-10).



## Project Structure and Key Functions

The `cnn.py` script is structured with the following key functions:

| Function Name              | Description                                                                  |
| :------------------------- | :--------------------------------------------------------------------------- |
| `get_data_loaders()`       | Loads and preprocesses the CIFAR-10 dataset.                               |
| `get_sample_images()`      | Retrieves a batch of sample images and labels.                               |
| `visualize(n)`             | Displays `n` images with their labels.                                       |
| `CNN(nn.Module)`           | Defines the Convolutional Neural Network architecture.                       |
| `define_loss_and_optimizer()` | Sets up the loss function and optimizer.                                     |
| `train_model()`            | Trains the model and plots the training loss.                                |
| `test_model()`             | Evaluates the model's accuracy on the given dataset.                         |
| `main` block (`if __name__ == "__main__":`) | Runs the complete training and testing pipeline.  
