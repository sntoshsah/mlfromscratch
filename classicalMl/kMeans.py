import numpy as np


class KMEANS:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        n_samples, n_features = X.shape
        # Randomly initialize centroids
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            # Assign clusters
            labels = self._assign_clusters(X)

            # Compute new centroids
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            print(f"Iteration {i+1}, Centroids:\n{new_centroids}")
            visualize_path = self.visualize_clusters(X, save_path=f"static/test_output/kmeans/kmeans_iteration_{i+1}.png")
            print(f"Cluster visualization saved to: {visualize_path}")

            # Check for convergence
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break

            self.centroids = new_centroids

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def predict(self, X):
        return self._assign_clusters(X)
    
    def inertia(self, X):
        labels = self._assign_clusters(X)
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            inertia += np.sum((cluster_points - self.centroids[k]) ** 2)
        return inertia
    
    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
    
    def get_centroids(self):
        return self.centroids
    
    def silhouette_score(self, X):
        labels = self._assign_clusters(X)
        n_samples = X.shape[0]
        silhouette_scores = np.zeros(n_samples)

        for i in range(n_samples):
            same_cluster = X[labels == labels[i]]
            other_clusters = [X[labels == k] for k in range(self.n_clusters) if k != labels[i]]

            a = np.mean(np.linalg.norm(same_cluster - X[i], axis=1)) if len(same_cluster) > 1 else 0
            b = np.min([np.mean(np.linalg.norm(cluster - X[i], axis=1)) for cluster in other_clusters]) if other_clusters else 0

            silhouette_scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0

        return np.mean(silhouette_scores)
    
    def elbow_method(self, X, max_k=10):
        inertias = []
        for k in range(1, max_k + 1):
            kmeans = KMEANS(n_clusters=k, max_iters=self.max_iters, tol=self.tol)
            kmeans.fit(X)
            inertias.append(kmeans.inertia(X))
        return inertias
    
    def visualize_clusters(self, X, save_path="static/test_output/kmeans_clusters.png"):
        import matplotlib.pyplot as plt

        if X.shape[1] != 2:
            raise ValueError("Visualization is only supported for 2D data.")

        labels = self._assign_clusters(X)

        plt.figure(figsize=(8, 6))
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {k}')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], s=300, c='black', marker='X', label='Centroids')
        plt.title('K-Means Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.savefig(save_path)
        plt.close()
        return save_path
    
if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    # Generate synthetic data
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # Fit KMeans
    kmeans = KMEANS(n_clusters=4)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    # Plot results
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centroids = kmeans.get_centroids()
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title('K-Means Clustering')
    plt.show()

    # Visualize clusters using the method
    kmeans.visualize_clusters(X)

    # Elbow method
    inertias = kmeans.elbow_method(X, max_k=10)
    plt.plot(range(1, 11), inertias, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Inertia')
    plt.savefig('static/test_output/kmeans/elbow_method.png')
    plt.close()
    # Print centroids
    print("Final Centroids:\n", kmeans.get_centroids())

    # Visualize final clusters
    kmeans.visualize_clusters(X, save_path="static/test_output/kmeans/kmeans_final.png")

    # Silhouette score
    score = kmeans.silhouette_score(X)
    print(f'Silhouette Score: {score}')

