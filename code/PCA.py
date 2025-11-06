import inspect
import numpy as np
from collections import Counter


"""
PCA (Principal Component Analysis) implementation for dimensionality reduction.
Algorithm:
1. Standardize the data (mean centering).
2. Compute the covariance matrix of the data.
3. Perform eigen decomposition on the covariance matrix to obtain eigenvalues and eigenvectors.
4. Sort eigenvalues and select the top 'n_components' eigenvectors.
5. Project the original data onto the selected eigenvectors to obtain the reduced dataset.


"""

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and corresponding eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top n_components
        self.components = sorted_eigenvectors[:, :self.n_components]

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def explained_variance(self):
        if self.components is None:
            raise ValueError("PCA not fitted yet.")
        return np.var(np.dot(self.components.T, self.components), axis=1)
    
    def explained_variance_ratio(self):
        if self.components is None:
            raise ValueError("PCA not fitted yet.")
        total_variance = np.sum(np.var(np.dot(self.components.T, self.components), axis=1))
        return np.var(np.dot(self.components.T, self.components), axis=1) / total_variance
    
    def reconstruct(self, X_transformed):
        return np.dot(X_transformed, self.components.T) + self.mean
    
    def visualize_components(self, X=None, y=None, save_path="static/test_output/pca_components.png"):
        import matplotlib.pyplot as plt

        if self.components is None:
            raise ValueError("PCA not fitted yet.")

        # Try to find X and y from caller if not provided
        if X is None:
            try:
                caller_globals = inspect.stack()[1].frame.f_globals
                X = caller_globals.get("X", None)
                if y is None:
                    y = caller_globals.get("y", None)
            except Exception:
                X = None

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        n_plots = 2 if (X is not None) else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        # Plot components (values across original feature indices)
        ax_comp = axes[0]
        feature_idx = np.arange(self.components.shape[0])
        for i in range(self.n_components):
            ax_comp.plot(feature_idx, self.components[:, i], marker='o', label=f'Component {i+1}')
        ax_comp.set_title('PCA Components')
        ax_comp.set_xlabel('Feature Index')
        ax_comp.set_ylabel('Component Value')
        ax_comp.legend()
        ax_comp.grid(True)

        # If data available, visualize its projection onto components
        if X is not None:
            X = np.asarray(X)
            try:
                X_trans = self.transform(X)
            except Exception as e:
                raise ValueError(f"Failed to transform provided X: {e}")

            ax_data = axes[1]
            if self.n_components >= 2:
                scatter = ax_data.scatter(X_trans[:, 0], X_trans[:, 1], c=y if y is not None else "C0", cmap='viridis', s=30, alpha=0.8)
                ax_data.set_xlabel('PC 1')
                ax_data.set_ylabel('PC 2')
                ax_data.set_title('Data projected onto first two principal components')
                if y is not None:
                    # add legend for discrete labels
                    try:
                        labels = np.unique(y)
                        handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i / max(1, len(labels)-1)), markersize=8) for i in range(len(labels))]
                        ax_data.legend(handles, labels, title="Label")
                    except Exception:
                        pass
            else:
                ax_data.hist(X_trans[:, 0], bins=30, color='C0', alpha=0.8)
                ax_data.set_title('Distribution of data on first principal component')
                ax_data.set_xlabel('PC 1')
                ax_data.set_ylabel('Count')

            ax_data.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()


if __name__ == "__main__":
    from sklearn.datasets import load_wine
    from matplotlib.lines import Line2D
    import os
    data = load_wine()
    X = data.data
    y = data.target

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    print("Original shape:", X.shape)
    print("Reduced shape:", X_reduced.shape)
    print("Explained variance ratio:", pca.explained_variance_ratio())
    X_reconstructed = pca.reconstruct(X_reduced)
    print("Reconstructed shape:", X_reconstructed.shape)
    print("Reconstruction error (MSE):", np.mean((X - X_reconstructed) ** 2))
    pca.visualize_components()
    

