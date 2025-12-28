## Run on Toy Dataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

# Load the dataset
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.images, digits.target, test_size=0.2, random_state=42
)
X_data = X_train  # Shape (1797, 8, 8)
y_data = y_train  # Shape (1797,)


# Normalize pixel values to [0, 1]
X_data = X_data / 16.0

# Convert labels to One-Hot Encoding
def one_hot(label, depth=10):
    ohl = np.zeros((1, depth))
    ohl[0, label] = 1
    return ohl
