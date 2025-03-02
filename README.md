# Deep Learning Neural Network for KMNIST Classification

## Overview
This project implements a fully connected neural network (MLP) from scratch in NumPy to classify the **Kuzushiji-MNIST (KMNIST)** dataset. The model supports various optimizers, including **SGD, Momentum, Nesterov, RMSprop, and Adam**.

## Features
- Fully connected neural network (MLP) implemented in NumPy.
- Supports different activation functions: **ReLU, Softmax**.
- Cross-entropy loss function.
- Multiple optimization algorithms: **SGD, Momentum, Nesterov, RMSprop, Adam**.
- Training and validation with accuracy and loss tracking.
- Image visualization with true and predicted labels.

## Dataset
[Kuzushiji-MNIST (KMNIST)](https://github.com/rois-codh/kmnist) is a dataset similar to MNIST but consists of 10 classes of Japanese characters. It includes:
- **Training set:** 60,000 images
- **Test set:** 10,000 images

## Installation
### Requirements
Ensure you have Python installed along with the following dependencies:
```sh
pip install numpy matplotlib
```

## Usage
### 1. Clone the repository
```sh
git clone https://github.com/vignesh6126/deep_learning.git
cd deep_learning
```

### 2. Download the KMNIST dataset
Download the following files from the [KMNIST website](http://codh.rois.ac.jp/kmnist/index.html.en) and place them in the project directory:
- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

### 3. Run the model training
```sh
python train.py
```

## Model Architecture
- **Input Layer:** 784 neurons (28x28 pixels, flattened)
- **Hidden Layers:** 512, 256 neurons with ReLU activation
- **Output Layer:** 10 neurons (Softmax activation for classification)
- **Optimizer:** Adam (default, can be changed)
- **Loss Function:** Cross-Entropy Loss

## Training and Evaluation
- **Training Data:** 80% of the dataset.
- **Validation Data:** 20% of the training set.
- **Batch Size:** 128
- **Epochs:** 10 (can be modified)

## Results
After training, the model evaluates its performance on the test set and displays the accuracy:
```sh
Test Accuracy: 0.92 
```

Additionally, sample test images with their true and predicted labels are displayed.


