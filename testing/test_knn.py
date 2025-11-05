import sys
import os

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from code.knn import KNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

def test_knn_iris(k=3):
    # Load the iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create KNN classifier instance
    knn = KNN(k=k)

    # Fit the model
    knn.fit(X_train, y_train)

    # Make predictions
    predictions = knn.predict(X_test)
    print(f'Predictions: {predictions}')

    # Calculate accuracy
    accuracy = np.sum(predictions == y_test) / len(y_test)
    print(f'KNN Classifier Accuracy on Iris Dataset: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    test_knn_iris(k=3)