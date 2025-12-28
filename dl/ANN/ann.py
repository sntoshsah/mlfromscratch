import numpy as np
from datasetup import dataset_setup
from graph import visualize_learning_curves

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return x * (1 - x)

# Initialize dataset and network parameters
X_train, y_train, X_test, y_test, input_nodes, hidden_nodes, output_nodes, W1, W2 = dataset_setup()
print("Weights and biases initialized for the ANN with bias terms.")


# Initialize biases
b1 = np.random.uniform(size=(1, hidden_nodes))
b2 = np.random.uniform(size=(1, output_nodes))

# Learning rate
lr = 0.01
epochs = 1000

# Lists to store metrics for plotting
train_accuracies = []
test_accuracies = []
train_losses = []
test_losses = []

# Helper function to calculate loss
def calculate_loss(y_true, y_pred):
    # Using binary cross-entropy for binary classification
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Helper function to evaluate the model
def evaluate_model(X_data, y_data, W1, b1, W2, b2):
    # Forward pass
    layer1 = sigmoid(np.dot(X_data, W1) + b1)
    output = sigmoid(np.dot(layer1, W2) + b2)
    
    predictions = (output > 0.5).astype(int)
    accuracy = np.mean(predictions == y_data) * 100
    loss = calculate_loss(y_data, output)
    return accuracy, loss

print("Starting ANN training with biases and metric tracking...")

for epoch in range(epochs):
    # --- Forward Pass --- (Training data)
    # Layer 1 (Hidden Layer)
    layer1_input = np.dot(X_train, W1) + b1
    layer1_output = sigmoid(layer1_input)

    # Output Layer
    output_layer_input = np.dot(layer1_output, W2) + b2
    output = sigmoid(output_layer_input)

    # --- Backpropagation ---
    # Output layer error and delta
    error = y_train - output
    d_output = error * sigmoid_derivative(output)

    # Hidden layer error and delta
    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(layer1_output)

    # Update Weights and Biases
    W2 += layer1_output.T.dot(d_output) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr
    W1 += X_train.T.dot(d_hidden) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

    # --- Evaluation and Metric Storage ---
    if epoch % 50 == 0 or epoch == epochs - 1:
        # Training metrics
        train_acc, train_loss = evaluate_model(X_train, y_train, W1, b1, W2, b2)
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)

        # Testing metrics
        test_acc, test_loss = evaluate_model(X_test, y_test, W1, b1, W2, b2)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        
        print(f"Epoch {epoch}, Train Acc: {train_acc:.2f}%, Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}%, Test Loss: {test_loss:.4f}")

print("Training complete.")

# Visualize learning curves
visualize_learning_curves(train_accuracies, test_accuracies, train_losses, test_losses, epochs)