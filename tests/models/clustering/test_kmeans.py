import pytest
import numpy as np
from nevula.core.tensor import Tensor
from nevula.models.clustering.kmeans import KMeans, KMeansPlusPlus
from nevula.models import get_model


def test_kmeans_initialization():
    km = KMeans(n_clusters=3, init="k-means++", random_state=42)
    assert km.n_clusters == 3
    assert km.init == "k-means++"
    assert not km.is_fitted


def test_kmeans_plus_plus_subclass():
    km_pp = KMeansPlusPlus(n_clusters=4, random_state=123)
    assert km_pp.n_clusters == 4
    assert km_pp.init == "k-means++"


def test_kmeans_synthetic_clusters():
    np.random.seed(42)
    # Generate 3 distinct clusters in 2D
    c1 = np.random.randn(30, 2) + np.array([0.0, 10.0])
    c2 = np.random.randn(30, 2) + np.array([10.0, 0.0])
    c3 = np.random.randn(30, 2) + np.array([-10.0, -10.0])
    X = np.vstack([c1, c2, c3])

    km = KMeans(n_clusters=3, init="k-means++", n_init=5, random_state=42)
    km.fit(X)

    assert km.is_fitted
    assert km.cluster_centers_ is not None
    assert km.cluster_centers_.shape == (3, 2)
    assert km.labels_ is not None
    assert km.labels_.shape == (90,)
    assert km.inertia_ > 0.0
    assert km.n_iter_ > 0

    # Predictions
    preds = km.predict(X)
    assert isinstance(preds, Tensor)
    assert preds.shape == (90,)
    
    # Check that each synthetic blob mostly has the same predicted cluster
    labels_np = preds.numpy()
    c1_labels = labels_np[:30]
    c2_labels = labels_np[30:60]
    c3_labels = labels_np[60:]

    assert len(np.unique(c1_labels)) == 1
    assert len(np.unique(c2_labels)) == 1
    assert len(np.unique(c3_labels)) == 1
    assert len(set([c1_labels[0], c2_labels[0], c3_labels[0]])) == 3


def test_kmeans_random_init():
    np.random.seed(42)
    X = np.random.randn(50, 4)
    km = KMeans(n_clusters=2, init="random", n_init=3, random_state=42)
    km.fit(X)

    assert km.is_fitted
    assert km.cluster_centers_.shape == (2, 4)


def test_kmeans_transform_and_score():
    np.random.seed(42)
    X = np.random.randn(40, 3)
    km = KMeans(n_clusters=3, random_state=42)
    km.fit(X)

    transformed = km.transform(X)
    assert transformed.shape == (40, 3)
    assert np.all(transformed.numpy() >= 0.0)

    score = km.score(X)
    assert score <= 0.0  # Opposite of inertia


def test_kmeans_registry_lookup():
    model_cls = get_model("kmeans")
    assert model_cls is KMeans


def test_kmeans_invalid_params():
    with pytest.raises(ValueError):
        KMeans(n_clusters=0)

    with pytest.raises(ValueError):
        KMeans(max_iter=-1)

    with pytest.raises(ValueError):
        km = KMeans(n_clusters=10)
        km.fit(np.random.randn(5, 2))  # n_samples < n_clusters
