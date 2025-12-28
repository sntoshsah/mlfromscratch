import numpy as np
from collections import Counter
from decisionTree import DecisionTree, visualize_tree

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, sample_size=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        self.sample_size = self.sample_size or n_samples

        for _ in range(self.n_trees):
            indices = np.random.choice(n_samples, self.sample_size, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)

        y_pred = [Counter(tree_pred).most_common(1)[0][0] for tree_pred in tree_preds]
        return np.array(y_pred)

    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy
    
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    clf = RandomForest(n_trees=5, max_depth=10)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print("Random Forest Classifier")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    tree_viz = visualize_tree(clf.trees[0].tree)
    tree_viz.render("static/test_output/random_forest_tree", format="png", cleanup=True)

