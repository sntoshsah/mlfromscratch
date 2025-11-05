import numpy as np


"""
Naive Bayes Classifier Implementation

Naive Bayes is a probabilistic classifier based on Bayes' theorem with the assumption of feature independence.
This implementation supports Gaussian Naive Bayes for continuous features.
"""

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.variance = {}
        self.prior = {}

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.variance[c] = np.var(X_c, axis=0)
            self.prior[c] = X_c.shape[0] / n_samples

    def gaussian_probability(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.variance[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict(self, X):
        y_pred = []
        for x in X:
            class_probabilities = {}
            for c in self.classes:
                prior = np.log(self.prior[c])
                conditional = np.sum(np.log(self.gaussian_probability(c, x)))
                class_probabilities[c] = prior + conditional
            predicted_class = max(class_probabilities, key=class_probabilities.get)
            y_pred.append(predicted_class)
        return np.array(y_pred)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred) / len(y)
        return accuracy
    
    def confusion_matrix(self, X, y):
        y_pred = self.predict(X)
        matrix = np.zeros((len(self.classes), len(self.classes)), dtype=int)
        for true, pred in zip(y, y_pred):
            matrix[true][pred] += 1
        return matrix
    
    def precision(self, X, y):
        y_pred = self.predict(X)
        precisions = {}
        for c in self.classes:
            tp = np.sum((y == c) & (y_pred == c))
            fp = np.sum((y != c) & (y_pred == c))
            precisions[c] = tp / (tp + fp) if (tp + fp) > 0 else 0
        return precisions
    
    def recall(self, X, y):
        y_pred = self.predict(X)
        recalls = {}
        for c in self.classes:
            tp = np.sum((y == c) & (y_pred == c))
            fn = np.sum((y == c) & (y_pred != c))
            recalls[c] = tp / (tp + fn) if (tp + fn) > 0 else 0
        return recalls
    
    def f1_score(self, X, y):
        precisions = self.precision(X, y)
        recalls = self.recall(X, y)
        f1_scores = {}
        for c in self.classes:
            p = precisions[c]
            r = recalls[c]
            f1_scores[c] = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return f1_scores
    
    def get_params(self):
        return {
            'mean': self.mean,
            'variance': self.variance,
            'prior': self.prior
        }
    
    def set_params(self, params):
        self.mean = params.get('mean', self.mean)
        self.variance = params.get('variance', self.variance)
        self.prior = params.get('prior', self.prior)
        self.classes = np.array(list(self.mean.keys()))

    def summary(self, X, y):
        acc = self.accuracy(X, y)
        cm = self.confusion_matrix(X, y)
        prec = self.precision(X, y)
        rec = self.recall(X, y)
        f1 = self.f1_score(X, y)
        summary_dict = {
            'accuracy': acc,
            'confusion_matrix': cm,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        }
        return summary_dict
    
    def print_summary(self, X, y):
        summary_dict = self.summary(X, y)
        print(f"Accuracy: {summary_dict['accuracy'] * 100:.2f}%")
        print("Confusion Matrix:")
        print(summary_dict['confusion_matrix'])
        print("Precision per class:")
        for c, p in summary_dict['precision'].items():
            print(f" Class {c}: {p:.4f}")
        print("Recall per class:")
        for c, r in summary_dict['recall'].items():
            print(f" Class {c}: {r:.4f}")
        print("F1 Score per class:")
        for c, f1 in summary_dict['f1_score'].items():
            print(f" Class {c}: {f1:.4f}")


if __name__ == "__main__":    # Simple test case
    from sklearn.datasets import load_iris, make_classification
    from sklearn.model_selection import train_test_split

    # Load dataset

    dataset = make_classification(n_samples=1000, n_features=10, n_classes=5, n_informative=5, random_state=42)
    print("Using synthetic classification dataset")
    print("Dataset shape:", dataset[0].shape)
    X, y = dataset
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train model
    model = NaiveBayes()
    model.fit(X_train, y_train)

    # Print summary
    model.print_summary(X_test, y_test)
