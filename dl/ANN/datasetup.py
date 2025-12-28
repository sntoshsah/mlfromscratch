from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np

def dataset_setup():
    global X_train, y_train, X_test, y_test
    global input_nodes, hidden_nodes, output_nodes
    global W1, W2

    # 1. Generate make_moons dataset
    X_full, y_full = make_moons(n_samples=2000, noise=0.2, random_state=42)
    y_full = y_full.reshape(-1, 1) # Reshape y to be a column vector
    # 2. Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
    # 3. Update network dimensions
    input_nodes = X_train.shape[1] # Number of features in make_moons is 2
    output_nodes = 1 # Binary classification for make_moons
    hidden_nodes = 3 # Keep hidden nodes as before for now
    # Initialize weights with new dimensions
    W1 = np.random.uniform(size=(input_nodes, hidden_nodes))
    W2 = np.random.uniform(size=(hidden_nodes, output_nodes))
    print(f"Input nodes: {input_nodes}, Output nodes: {output_nodes}, Hidden nodes: {hidden_nodes}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print("Data setup complete for make_moons dataset.")

    return X_train, y_train, X_test, y_test, input_nodes, hidden_nodes, output_nodes, W1, W2

