import sys
import os

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from code.logistic_reg import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

def test_logistic_regression_breast_cancer():
    # Load the breast cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create Logistic Regression model instance
    log_reg = LogisticRegression()

    # Fit the model
    log_reg.fit(X_train, y_train)

    # Make predictions
    predictions = log_reg.predict(X_test)
    print(f'Predictions: {predictions}')

    # Calculate accuracy
    accuracy = log_reg.accuracy(X_test, y_test)
    print(f'Logistic Regression Accuracy on Breast Cancer Dataset: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    test_logistic_regression_breast_cancer()
