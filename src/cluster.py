
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple


def kmeans_states(features: np.ndarray,
                  n_clusters: int = 3,
                  random_state: int = 42) -> Tuple[np.ndarray, KMeans]:
    """
    Cluster feature vectors into n_clusters using KMeans.

    Parameters
    ----------
    features : np.ndarray
        Array of shape (n_samples, n_features).
    n_clusters : int
        Number of clusters to find (hypothesized states).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    labels : np.ndarray
        Cluster labels for each sample.
    model : KMeans
        Fitted KMeans model.
    """
    if features.ndim != 2:
        raise ValueError("features must be a 2D array (n_samples, n_features).")

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = model.fit_predict(features)
    return labels, model
