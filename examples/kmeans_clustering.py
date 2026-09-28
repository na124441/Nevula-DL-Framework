"""
K-Means and K-Means++ Clustering Example in Nevula Deep Learning Framework.
"""
import numpy as np
import nevula
from nevula.models.clustering.kmeans import KMeans, KMeansPlusPlus


def main():
    print("=" * 60)
    print(" Nevula Clustering: K-Means & K-Means++ Demo")
    print("=" * 60)

    # 1. Generate synthetic clustered dataset
    np.random.seed(42)
    n_samples_per_cluster = 50
    cluster_centers = [
        np.array([2.0, 2.0]),
        np.array([-3.0, 3.0]),
        np.array([0.0, -4.0]),
    ]

    data_parts = []
    for center in cluster_centers:
        cluster_data = np.random.randn(n_samples_per_cluster, 2) * 0.7 + center
        data_parts.append(cluster_data)
    X = np.vstack(data_parts)

    print(f"\nGenerated dataset of shape {X.shape} with 3 distinct clusters.")

    # 2. Fit K-Means with K-Means++ initialization
    print("\n--- Training KMeans (init='k-means++') ---")
    kmeans_pp = KMeansPlusPlus(n_clusters=3, n_init=10, random_state=42)
    kmeans_pp.fit(X)

    print(f"Converged in {kmeans_pp.n_iter_} iterations.")
    print(f"Final Inertia (WCSS): {kmeans_pp.inertia_:.4f}")
    print("Learned Centroids:")
    print(kmeans_pp.cluster_centers_.numpy())

    # 3. Fit K-Means with Random initialization
    print("\n--- Training KMeans (init='random') ---")
    kmeans_rnd = KMeans(n_clusters=3, init="random", n_init=10, random_state=42)
    kmeans_rnd.fit(X)

    print(f"Converged in {kmeans_rnd.n_iter_} iterations.")
    print(f"Final Inertia (WCSS): {kmeans_rnd.inertia_:.4f}")
    print("Learned Centroids:")
    print(kmeans_rnd.cluster_centers_.numpy())

    # 4. Predict cluster labels for new test points
    test_points = np.array([
        [2.1, 1.9],    # Close to cluster 1
        [-2.9, 3.2],   # Close to cluster 2
        [0.1, -4.2],   # Close to cluster 3
    ])
    predictions = kmeans_pp.predict(test_points)
    print("\nPredictions for sample test points:")
    for pt, label in zip(test_points, predictions.numpy()):
        print(f"  Point {pt} -> Cluster {label}")

    # 5. Transform points to cluster distance space
    distances = kmeans_pp.transform(test_points)
    print("\nDistances to cluster centers:")
    print(distances.numpy())

    print("\nRegistered Clustering Models in Nevula:")
    print(nevula.models.summary())
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
