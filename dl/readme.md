# Deep Learning from Scratch

This directory contains implementations of fundamental deep learning algorithms built entirely from scratch using NumPy. The goal is to provide educational examples that demonstrate the core concepts of neural networks without relying on high-level frameworks like TensorFlow or PyTorch.

## Modules

### Artificial Neural Network (ANN)
- **Location**: `ANN/`
- **Description**: A simple 2-layer feedforward neural network with sigmoid activations for binary classification. Includes manual forward pass, backpropagation, and learning curve visualization.
- **Main File**: `ann.py`
- **Usage**: Run `python dl/ANN/ann.py` from the repository root.
- **Details**: See [ANN/README.md](ANN/readme.md) for comprehensive documentation.

### Convolutional Neural Network (CNN)
- **Location**: `CNN/`
- **Description**: A basic convolutional neural network with convolution, ReLU, flatten, and dense layers for image classification. Demonstrates convolutional operations and backpropagation through the network.
- **Main File**: `trainer.py`
- **Usage**: Run `python dl/CNN/trainer.py` from the repository root.
- **Details**: See [CNN/README.md](CNN/readme.md) for comprehensive documentation.

### Recurrent Neural Network (RNN)
- **Location**: `RNN/`
- **Description**: A character-level RNN using tanh activations for sequence prediction. Implements backpropagation through time (BPTT) for training on sequential data.
- **Main File**: `trainer.py`
- **Usage**: Run `python dl/RNN/trainer.py` from the repository root.
- **Details**: See [RNN/README.md](RNN/readme.md) for comprehensive documentation.

## Requirements

All implementations require the following dependencies (install via `pip install -r requirements.txt` from the repository root):

- NumPy
- Matplotlib (for visualization)
- Scikit-learn (used in some dataset setups)

## Getting Started

1. Clone the repository and navigate to the root directory.
2. Install dependencies: `pip install -r requirements.txt`
3. Run any of the modules as described above.

Each module is self-contained and includes its own dataset setup, training loop, and visualization. The code is designed for educational purposes and prioritizes clarity over performance optimization.

## Project Structure

```
dl/
├── ANN/
│   ├── ann.py          # Main ANN implementation
│   ├── datasetup.py    # Dataset preparation
│   ├── graph.py        # Visualization utilities
│   └── readme.md       # Detailed ANN documentation
├── CNN/
│   ├── cnn.py          # CNN layer implementations
│   ├── datasetup.py    # Dataset preparation
│   ├── graph.py        # Visualization utilities
│   ├── trainer.py      # Main CNN training script
│   └── readme.md       # Detailed CNN documentation
├── RNN/
│   ├── rnn.py          # RNN implementation
│   ├── datasetup.py    # Dataset preparation
│   ├── graph.py        # Visualization utilities
│   ├── trainer.py      # Main RNN training script
│   └── readme.md       # Detailed RNN documentation
└── readme.md           # This file
```

## Learning Objectives

These implementations help understand:
- Forward and backward propagation
- Gradient descent optimization
- Activation functions and loss functions
- Convolutional operations
- Recurrent connections and sequence modeling
- Numerical stability considerations

