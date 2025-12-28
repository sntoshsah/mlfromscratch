import numpy as np

class LogisticRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, epochs=1000, lr=0.01):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        learning_rate = lr
        n_iterations = epochs

        for _ in range(n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias # Compute linear model wx + b
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) # Gradient calculation dw = (1/m) * \sum((y_predicted - y) * x_i)
            # dw = (1 / n_samples) * np.sum((y_predicted - y)[:, np.newaxis] * X, axis=0) # Gradient calculation dw = (1/m) * \sum((y_predicted - y) * x_i)
            db = (1 / n_samples) * np.sum(y_predicted - y) # Gradient calculation db = (1/m) * \sum(y_predicted - y)

            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred) / len(y)
        return accuracy