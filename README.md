# MNIST Neural Network from Scratch

## Overview
This project implements a fully connected neural network from scratch to classify handwritten digits from the MNIST dataset. The neural network is built using NumPy, without relying on high-level deep learning libraries such as TensorFlow or PyTorch. The goal is to understand the fundamental principles of neural networks, including forward propagation, backpropagation, and optimization.

## Features
- Implements a multi-layer perceptron (MLP) for digit classification.
- Uses NumPy for all matrix operations and calculations.
- Performs forward and backward propagation manually.
- Costumizable number of neurons per layer
- Trains the model using stochastic gradient descent (SGD) with momentum.
- Includes functions for weight initialization, activation functions, and loss computation.
- Evaluates model performance on the MNIST test dataset.

## Dataset
The project uses the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits (0-9). Each image is a 28x28 grayscale pixel array.

## Installation
### Prerequisites
Ensure you have Python installed along with the required libraries:
```bash
pip install numpy matplotlib
```

### Clone the Repository
```bash
git clone https://github.com/Leon-MRR/MNIST-NN-from-scratch.git
cd MNIST-NN-from-scratch
```

## Neural Network Architecture
The network consists of:
- Input Layer: 784 neurons (28x28 pixels)
- Hidden Layers: 2 layers with costumizable neuron values and ReLU activation
- Output Layer: 10 neurons (one for each digit, using softmax activation)

## Results
The model achieves a reasonable accuracy on the MNIST dataset, demonstrating the effectiveness of manually implemented neural networks.


## Contributing
Feel free to fork the repository and submit pull requests for improvements or additional features.



