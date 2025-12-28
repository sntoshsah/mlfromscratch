import numpy as np


class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1

    def fit(self, X, y, sample_weights):
        n_samples, n_features = X.shape
        min_error = float('inf')

        for feature_i in range(n_features):
            feature_values = X[:, feature_i]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    predictions[polarity * feature_values < polarity * threshold] = -1

                    misclassified = predictions != y
                    error = np.sum(sample_weights * misclassified)

                    if error < min_error:
                        min_error = error
                        self.polarity = polarity
                        self.threshold = threshold
                        self.feature_index = feature_i

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        feature_values = X[:, self.feature_index]
        predictions[self.polarity * feature_values < self.polarity * self.threshold] = -1
        return predictions

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.models = []
        self.model_weights = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights to 1/N
        sample_weights = np.full(n_samples, (1 / n_samples))

        for _ in range(self.n_estimators):
            # Train a weak learner (decision stump)
            model = DecisionStump()
            model.fit(X, y, sample_weights)
            predictions = model.predict(X)

            # Calculate error and model weight
            misclassified = predictions != y
            error = np.sum(sample_weights * misclassified) / np.sum(sample_weights)

            # Avoid division by zero
            if error == 0:
                model_weight = 1
            else:
                model_weight = 0.5 * np.log((1 - error) / (error + 1e-10))

            # Update sample weights
            sample_weights *= np.exp(-model_weight * y * predictions)
            sample_weights /= np.sum(sample_weights)  # Normalize

            # Save the model and its weight
            self.models.append(model)
            self.model_weights.append(model_weight)

    def predict(self, X):
        final_predictions = np.zeros(X.shape[0])
        for model, weight in zip(self.models, self.model_weights):
            final_predictions += weight * model.predict(X)
        return np.sign(final_predictions)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy
    

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Create a simple dataset
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=42)
    y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train AdaBoost
    model = AdaBoost(n_estimators=10)
    model.fit(X_train, y_train)

    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"AdaBoost Test Accuracy: {accuracy * 100:.2f}%")