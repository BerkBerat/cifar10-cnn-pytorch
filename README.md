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

## Model Architecture

Input: 3x32x32 color image

[ Conv2d(3, 32, kernel_size=3, padding=1) ]
→ ReLU
→ MaxPool2d(kernel_size=2, stride=2)

[ Conv2d(32, 64, kernel_size=3, padding=1) ]
→ ReLU
→ MaxPool2d(kernel_size=2, stride=2)

→ Flatten

[ Linear(64*8*8, 128) ]
→ ReLU
→ Dropout(p=0.2)

[ Linear(128, 10) ]
→ Output


cnn.py
├── get_data_loaders()           # Loads and preprocesses CIFAR-10
├── get_sample_images()         # Loads sample images and labels
├── visualize(n)                # Displays n images with labels
├── CNN(nn.Module)              # Defines CNN architecture
├── define_loss_and_optimizer() # Sets loss and optimizer
├── train_model()               # Trains the model and plots loss
├── test_model()                # Evaluates accuracy on data
└── __main__ block              # Runs training + testing pipeline
