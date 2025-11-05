import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Convert y to -1 and 1
        y_ = np.array([1 if i > 0 else -1 for i in y])

        # Training
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.sign(linear_output)

                # Update rule
                if y_predicted != y_[idx]:
                    self.weights += self.lr * y_[idx] * x_i
                    self.bias += self.lr * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = np.sign(linear_output)
        # Convert back to original labels 0 and 1
        y_predicted = np.array([1 if i > 0 else 0 for i in y_predicted])
        return y_predicted
    
    def accuracy(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred) / len(y)
        return accuracy
    
    
if __name__ == "__main__":
    # Simple test
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, n_informative=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
    perceptron.fit(X_train, y_train)
    accuracy = perceptron.accuracy(X_test, y_test)
    print(f'Perceptron Accuracy: {accuracy:.4f}')
