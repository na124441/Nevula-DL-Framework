from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from nevula.core.tensor import Tensor
from nevula.models.base import BaseModel
from nevula.models.registry import register_model


@register_model(name="kmeans", category="clustering")
class KMeans(BaseModel):
    """
    K-Means and K-Means++ Clustering Algorithm for the Nevula Deep Learning Framework.

    Mathematical Formulation:
        Given an unlabeled dataset X = {x_1, x_2, ..., x_N} in R^D and target clusters K:
        Find cluster centroids mu_1, ..., mu_K that minimize the inertia (Within-Cluster Sum of Squares):
            J = sum_{i=1}^N min_{k=1..K} ||x_i - mu_k||_2^2

        Lloyd's Iterative Algorithm:
            1. Assignment Step:
               c_i = argmin_{k} ||x_i - mu_k||_2^2
            2. Update Step:
               mu_k = (1 / |S_k|) sum_{i in S_k} x_i

        K-Means++ Initialization (Arthur & Vassilvitskii, 2007):
            1. Choose the first centroid mu_1 uniformly at random from X.
            2. For each point x, compute D(x) = min_{j < k} ||x - mu_j||_2.
            3. Choose the next centroid mu_k from X with probability proportional to D(x)^2.
            4. Repeat steps 2 and 3 until K centroids are chosen.

    Parameters:
        n_clusters: Number of clusters (K) to form as well as the number of centroids. Default: 8.
        init: Initialization method ('k-means++', 'random', or ndarray/Tensor of pre-computed centroids). Default: 'k-means++'.
        n_init: Number of times the k-means algorithm will be run with different centroid seeds.
                The final results will be the best output of n_init consecutive runs in terms of inertia. Default: 10.
        max_iter: Maximum number of iterations of the k-means algorithm for a single run. Default: 300.
        tol: Relative tolerance with regards to Frobenius norm of the difference in the cluster centers. Default: 1e-4.
        random_state: Seed or RandomState for reproducible centroid generation. Default: None.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: Union[str, np.ndarray, Tensor] = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        if n_clusters <= 0:
            raise ValueError(f"n_clusters must be > 0, got {n_clusters}")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be > 0, got {max_iter}")
        if tol < 0:
            raise ValueError(f"tol must be non-negative, got {tol}")

        self.n_clusters = int(n_clusters)
        self.init = init
        self.n_init = max(1, int(n_init))
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

        # Fitted attributes
        self.cluster_centers_: Optional[Tensor] = None
        self.labels_: Optional[Tensor] = None
        self.inertia_: float = float("inf")
        self.n_iter_: int = 0
        self.n_features_in_: Optional[int] = None

    def _init_centroids(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Initializes cluster centroids according to `self.init`."""
        n_samples, n_features = X.shape

        if isinstance(self.init, (np.ndarray, Tensor)):
            centers = self.init.numpy() if isinstance(self.init, Tensor) else np.asarray(self.init, dtype=np.float64)
            if centers.shape != (self.n_clusters, n_features):
                raise ValueError(
                    f"Given init centroids shape {centers.shape} does not match (n_clusters, n_features)=({self.n_clusters}, {n_features})"
                )
            return centers.copy()

        init_method = str(self.init).lower().strip().replace("-", "_").replace(" ", "_")

        if init_method in ("k_means++", "kmeans++", "plus_plus", "kmeans_plus_plus"):
            # K-Means++ initialization
            centers = np.empty((self.n_clusters, n_features), dtype=np.float64)
            # Step 1: Pick first center uniformly at random
            first_idx = rng.randint(0, n_samples)
            centers[0] = X[first_idx]

            # Closest squared distances
            closest_dist_sq = np.sum((X - centers[0]) ** 2, axis=1)

            for c in range(1, self.n_clusters):
                # Step 2: Probability proportional to D(x)^2
                dist_sum = np.sum(closest_dist_sq)
                if dist_sum > 0:
                    probs = closest_dist_sq / dist_sum
                    candidate_idx = rng.choice(n_samples, p=probs)
                else:
                    candidate_idx = rng.randint(0, n_samples)

                centers[c] = X[candidate_idx]

                # Update squared distances with the new centroid
                new_dist_sq = np.sum((X - centers[c]) ** 2, axis=1)
                closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

            return centers

        elif init_method in ("random", "uniform"):
            # Uniform random sampling of points from dataset without replacement
            indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
            return X[indices].copy()

        else:
            raise ValueError(
                f"Unknown init scheme '{self.init}'. Expected 'k-means++', 'random', or ndarray/Tensor."
            )

    def _single_run(
        self, X: np.ndarray, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """Executes a single Lloyd's K-Means clustering run."""
        n_samples, n_features = X.shape
        centers = self._init_centroids(X, rng)

        labels = np.zeros(n_samples, dtype=np.int64)
        inertia = float("inf")
        iteration = 0

        for it in range(self.max_iter):
            iteration = it + 1
            # Assignment Step: compute pairwise squared Euclidean distance
            # ||X - C||^2 = ||X||^2 - 2 X C^T + ||C||^2
            dists = np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
            new_labels = np.argmin(dists, axis=1)

            # Calculate current inertia (sum of squared distances to closest center)
            min_dists = np.min(dists, axis=1)
            current_inertia = float(np.sum(min_dists))

            # Update Step: recompute cluster centers
            new_centers = np.empty_like(centers)
            for k in range(self.n_clusters):
                cluster_mask = new_labels == k
                if np.any(cluster_mask):
                    new_centers[k] = np.mean(X[cluster_mask], axis=0)
                else:
                    # Handle empty cluster: re-seed from the point furthest away from any center
                    furthest_idx = np.argmax(min_dists)
                    new_centers[k] = X[furthest_idx]

            # Check convergence via Frobenius norm shift
            center_shift = np.sqrt(np.sum((new_centers - centers) ** 2))
            centers = new_centers
            labels = new_labels
            inertia = current_inertia

            if center_shift <= self.tol:
                break

        return centers, labels, inertia, iteration

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> "KMeans":
        """
        Computes k-means clustering.

        Args:
            X: Input training data of shape (n_samples, n_features) as Tensor, numpy array or sequence.
            y: Ignored (present for scikit-learn / BaseModel API compatibility).
            **kwargs: Extra arguments.

        Returns:
            self: The fitted KMeans model.
        """
        X_tensor = self._to_tensor(X)
        X_np = np.asarray(X_tensor.numpy(), dtype=np.float64)

        if X_np.ndim != 2:
            raise ValueError(f"Expected 2D input array (n_samples, n_features), got shape {X_np.shape}")

        n_samples, n_features = X_np.shape
        if n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples={n_samples} should be >= n_clusters={self.n_clusters}."
            )

        self.n_features_in_ = n_features
        base_seed = self.random_state if self.random_state is not None else np.random.randint(0, 1000000)

        best_centers: Optional[np.ndarray] = None
        best_labels: Optional[np.ndarray] = None
        best_inertia = float("inf")
        best_n_iter = 0

        # Run n_init times and keep the one with lowest inertia
        num_runs = 1 if isinstance(self.init, (np.ndarray, Tensor)) else self.n_init
        for run_idx in range(num_runs):
            run_rng = np.random.RandomState(base_seed + run_idx)
            centers, labels, inertia, iters = self._single_run(X_np, run_rng)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
                best_n_iter = iters

        self.cluster_centers_ = Tensor(best_centers)
        self.labels_ = Tensor(best_labels)
        self.inertia_ = float(best_inertia)
        self.n_iter_ = best_n_iter
        self._is_fitted = True
        return self

    def forward(self, X: Tensor) -> Tensor:
        """
        Computes the squared Euclidean distances from X to each cluster center.

        Args:
            X: Input Tensor of shape (batch_size, n_features).

        Returns:
            Tensor: Pairwise squared Euclidean distances of shape (batch_size, n_clusters).
        """
        if not self._is_fitted or self.cluster_centers_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit() before calling forward().")

        X_np = np.asarray(X.numpy(), dtype=np.float64)
        centers_np = self.cluster_centers_.numpy()

        # Pairwise squared Euclidean distance: (batch_size, n_clusters)
        dists = np.sum((X_np[:, np.newaxis, :] - centers_np[np.newaxis, :, :]) ** 2, axis=2)
        return Tensor(dists)

    def predict(self, X: Any) -> Tensor:
        """
        Predicts the closest cluster each sample in X belongs to.

        Args:
            X: New data of shape (n_samples, n_features).

        Returns:
            Tensor: Cluster labels for each sample of shape (n_samples,).
        """
        if not self._is_fitted or self.cluster_centers_ is None:
            raise RuntimeError("This KMeans instance is not fitted yet. Call fit() first.")

        X_tensor = self._to_tensor(X)
        distances = self.forward(X_tensor)
        labels = np.argmin(distances.numpy(), axis=1)
        return Tensor(labels)

    def fit_predict(self, X: Any, y: Any = None) -> Tensor:
        """
        Computes cluster centers and predicts cluster index for each sample.

        Args:
            X: Input samples.

        Returns:
            Tensor: Cluster labels for each sample.
        """
        return self.fit(X, y).predict(X)

    def transform(self, X: Any) -> Tensor:
        """
        Transforms X to a cluster-distance space.
        In the new space, each dimension is the distance to the cluster center.

        Args:
            X: Input samples of shape (n_samples, n_features).

        Returns:
            Tensor: Euclidean distances of shape (n_samples, n_clusters).
        """
        sq_dists = self.forward(self._to_tensor(X))
        return Tensor(np.sqrt(np.maximum(sq_dists.numpy(), 0.0)))

    def fit_transform(self, X: Any, y: Any = None) -> Tensor:
        """
        Fits to data, then transforms it into cluster-distance space.
        """
        return self.fit(X, y).transform(X)

    def score(self, X: Any, y: Any = None) -> float:
        """
        Opposite of the value of X on the K-means objective (negative inertia).

        Args:
            X: Input samples.

        Returns:
            float: Negative inertia on X.
        """
        X_tensor = self._to_tensor(X)
        distances = self.forward(X_tensor).numpy()
        return -float(np.sum(np.min(distances, axis=1)))

    def get_config(self) -> Dict[str, Any]:
        """Returns hyperparameter configuration dictionary."""
        return {
            "n_clusters": self.n_clusters,
            "init": self.init if isinstance(self.init, str) else "custom_array",
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
        }


# Convenience alias for explicitly referencing KMeans++
class KMeansPlusPlus(KMeans):
    """
    K-Means with K-Means++ centroid initialization strategy.
    Equivalent to KMeans(..., init='k-means++').
    """

    def __init__(
        self,
        n_clusters: int = 8,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )


__all__ = ["KMeans", "KMeansPlusPlus"]
