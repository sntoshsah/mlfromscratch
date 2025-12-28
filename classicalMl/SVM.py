import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)  # Convert labels to -1 and 1

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    dw = 2 * self.lambda_param * self.weights
                    db = 0
                else:
                    dw = 2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx])
                    db = y_[idx]

                self.weights -= self.lr * dw
                self.bias -= self.lr * db

    def predict(self, X):
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)
    
    def accuracy(self, X, y):
        y_pred = self.predict(X)
        y_ = np.where(y <= 0, -1, 1)
        accuracy = np.sum(y_ == y_pred) / len(y)
        return accuracy
    
    def visualize(self, X, y):
        import matplotlib.pyplot as plt
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)

        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        x = np.linspace(xlim[0], xlim[1], 30)
        y_decision = get_hyperplane_value(x, self.weights, self.bias, 0)
        y_positive = get_hyperplane_value(x, self.weights, self.bias, 1)
        y_negative = get_hyperplane_value(x, self.weights, self.bias, -1)

        plt.plot(x, y_decision, 'k-')
        plt.plot(x, y_positive, 'k--')
        plt.plot(x, y_negative, 'k--')

        plt.fill_between(x, y_positive, y_negative, color='grey', alpha=0.2)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.savefig('static/test_output/svm_visualization.png')
        plt.show()
        plt.close()

if __name__ == "__main__":
    # Simple test
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split

    X, y = make_blobs(n_samples=1000, centers=7, n_features=2, random_state=42)
    y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm.fit(X_train, y_train)
    accuracy = svm.accuracy(X_test, y_test)
    print(f'SVM Accuracy: {accuracy:.4f}')
    svm.visualize(X, y)