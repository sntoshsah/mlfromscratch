import sys
import os

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from code.lin_reg import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np

def test_linear_regression_boston():
    # Load the Boston housing dataset
    X, y = make_regression(n_samples=100, n_features=13, noise=0.1, random_state=42)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create Linear Regression model instance
    lin_reg = LinearRegression()

    # Fit the model
    lin_reg.fit(X_train, y_train)

    # Make predictions
    predictions = lin_reg.predict(X_test)
    print(f'Predictions: {predictions}')

    # Calculate R^2 score
    r2 = lin_reg.r2_score(X_test, y_test)
    print(f'Linear Regression R^2 Score on Boston Dataset: {r2:.4f}')

if __name__ == "__main__":
    test_linear_regression_boston()