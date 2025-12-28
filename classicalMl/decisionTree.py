import numpy as np

"""A simple Decision Tree classifier implementation.
Algorithm:
1. Start at the root node and consider all features.
2. For each feature, find the best threshold to split the data to maximize information gain.
3. Split the dataset into left and right branches based on the best feature and threshold.
4. Recursively repeat the process for each branch until a stopping criterion is met (e.g., maximum depth, pure leaf nodes).
5. Assign class labels to leaf nodes based on the majority class of the samples in that node.
        (n_left / n) * self.entropy(y[left_indices]) + (n_right / n) * self.entropy(y[right_indices])
        return parent_entropy - child_entropy
"""

def entropy(self, y):
    class_labels, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
    return entropy

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        predictions = [self._predict_sample(sample, self.tree) for sample in X]
        return np.array(predictions)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        # Stopping criteria
        if len(unique_classes) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return unique_classes[np.argmax(class_counts)]
        
        # Find the best split
        best_feature, best_threshold = self._best_split(X, y, num_features)
        if best_feature is None:
            return unique_classes[np.argmax(class_counts)]
        
        # Create subtrees
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold
        
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _best_split(self, X, y, num_features):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold

    def _information_gain(self, X, y, feature, threshold):
        parent_entropy = self._entropy(y)
        
        left_indices = X[:, feature] < threshold
        right_indices = X[:, feature] >= threshold
        
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0
        
        n = len(y)
        n_left, n_right = len(y[left_indices]), len(y[right_indices])

        child_entropy = (n_left / n) * self._entropy(y[left_indices]) + (n_right / n) * self._entropy(y[right_indices])
        gain = parent_entropy - child_entropy
        # print(f"Information Gain for feature {feature} with threshold {threshold}: {gain}")
        return gain

    def _entropy(self, y):
        class_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
        return entropy

    def _predict_sample(self, sample, tree):
        if not isinstance(tree, tuple):
            return tree

        feature, threshold, left_subtree, right_subtree = tree
        if sample[feature] < threshold:
            return self._predict_sample(sample, left_subtree)
        else:
            return self._predict_sample(sample, right_subtree)
        
    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy
    
    def print_tree(self, tree=None, depth=0):
        if tree is None:
            tree = self.tree

        if not isinstance(tree, tuple):
            print("\t" * depth + f"Leaf_{tree}")
            return

        feature, threshold, left_subtree, right_subtree = tree
        print("\t" * depth + f"[X{feature} < {threshold}]")
        self.print_tree(left_subtree, depth + 1)
        self.print_tree(right_subtree, depth + 1)



from matplotlib import pyplot as plt
from graphviz import Digraph

def visualize_tree(tree, feature_names=None, class_names=None):
    dot = Digraph()
    
    def add_nodes_edges(tree, parent_name=None, edge_label=""):
        if not isinstance(tree, tuple):
            node_name = f"Leaf_{tree}"
            dot.node(node_name, node_name, shape='box', style='filled', color='lightgrey')
            if parent_name:
                dot.edge(parent_name, node_name, label=edge_label)
            return

        feature, threshold, left_subtree, right_subtree = tree
        node_name = f"[X{feature} < {threshold}]"
        dot.node(node_name, node_name)

        if parent_name:
            dot.edge(parent_name, node_name, label=edge_label)

        add_nodes_edges(left_subtree, node_name, "True")
        add_nodes_edges(right_subtree, node_name, "False")

    add_nodes_edges(tree)
    return dot

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTree(max_depth=3)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    accuracy = np.sum(predictions == y_test) / len(y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    clf.print_tree()
    tree_viz = visualize_tree(clf.tree)
    tree_viz.render("static/test_output/decision_tree", format="png", cleanup=True)


    
