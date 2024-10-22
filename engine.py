import numpy as np
import pandas as pd
import time 
from sklearn.datasets import make_blobs, make_moons, make_circles, make_swiss_roll

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-6, metric='euclidean', init_strategy='random', method='kmeans'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.metric = metric
        self.init_strategy = init_strategy
        self.method = method
        self.centroids = None
        self.iter_centroids = []
        self.labels = None
        self.inertia_ = None

    def _calculate_distances(self, X, Y):

        X = np.array(X)
        Y = np.array(Y)

        if self.metric == 'euclidean':
            return np.linalg.norm(X[:, np.newaxis] -Y, axis=2)
        elif self.metric == 'manhattan':
            return np.sum(np.abs(X[:, np.newaxis] - Y), axis=2)
        elif self.metric == 'cosine':
            dot_product = np.dot(X, Y.T) 

            norm_X = np.linalg.norm(X, axis=1)[:, np.newaxis] 
            norm_Y = np.linalg.norm(Y, axis=1) 
            
            cos_similarity = dot_product / (norm_X * norm_Y + 1e-10)
            return 1 - cos_similarity 
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _initialize_centroids(self, X):
        if self.init_strategy == 'random':
            random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            return X[random_indices]
        elif self.init_strategy == 'kmeans++':
            return self._kmeans_plus_plus(X)
        else:
            raise ValueError("Invalid init_strategy.")

    def _kmeans_plus_plus(self, X):
        self.centroids = [X[np.random.choice(X.shape[0])]] 
        for _ in range(1, self.n_clusters):
            distances = np.min(self._calculate_distances(X, self.centroids), axis=1)
            probabilities = distances ** 2 / np.sum(distances ** 2)
            next_centroid_idx = np.random.choice(X.shape[0], p=probabilities)
            self.centroids.append(X[next_centroid_idx])
        return np.array(self.centroids)

    def fit(self, X):
        self.centroids = self._initialize_centroids(X)
        for _ in range(self.max_iter):
            distances = self._calculate_distances(X, self.centroids)
            self.labels = np.argmin(distances, axis=1)

            if self.method == 'kmeans':
                new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])
            elif self.method == 'kmedians':
                new_centroids = np.array([np.median(X[self.labels == i], axis=0) for i in range(self.n_clusters)])
            elif self.method == 'kmedoids':
                new_centroids = np.array([self._calculate_medoid(X, self.labels, i) for i in range(self.n_clusters)])

            self.iter_centroids.append(new_centroids.copy())
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            self.centroids = new_centroids

        self.inertia_ = self._calculate_inertia(X)
        return self.labels

    def _calculate_medoid(self, X, labels, cluster_id):
        cluster_points = X[labels == cluster_id]
        distances = self._calculate_distances(cluster_points, cluster_points)
        return cluster_points[np.argmin(np.sum(distances, axis=1))]

    def _calculate_inertia(self, X):
        distances = np.linalg.norm(X - self.centroids[self.labels], axis=1)
        return np.sum(distances ** 2)



def silhouette_score(X, labels, metric='euclidean'):
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    
    if len(unique_labels) < 2:
        return np.nan  
    
    silhouette_scores = []

    for i in range(n_samples):
        current_point = X[i]
        current_label = labels[i]

        same_cluster = X[labels == current_label]
        if len(same_cluster) > 1: 
            a_i = np.mean([calculate_pairwise_distance(current_point, p, metric) 
                           for p in same_cluster if not np.array_equal(current_point, p)])
        else:
            a_i = 0  

        b_i = np.inf
        for label in unique_labels:
            if label == current_label:
                continue

            other_cluster = X[labels == label]
            b_i_candidate = np.mean([calculate_pairwise_distance(current_point, p, metric) 
                                     for p in other_cluster])
            b_i = min(b_i, b_i_candidate)

        s_i = (b_i - a_i) / max(a_i, b_i) 
        silhouette_scores.append(s_i)

    return np.mean(silhouette_scores) if silhouette_scores else np.nan



def calculate_intra_cluster_distance(cluster_points):
    if len(cluster_points) == 0:
        return 0
    distances = np.linalg.norm(cluster_points[:, np.newaxis] - cluster_points, axis=2)
    return np.mean(distances)

def calculate_inter_cluster_distance(centroid1, centroid2):
    return np.linalg.norm(centroid1 - centroid2)

def davies_bouldin_score(X, labels):

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    centroids = np.array([X[labels == label].mean(axis=0) for label in unique_labels])
    
    intra_distances = np.array([calculate_intra_cluster_distance(X[labels == label]) for label in unique_labels])
    
    db_index = 0.0
    
    for i in range(n_clusters):
        max_ratio = 0.0
        for j in range(n_clusters):
            if i != j:
                inter_distance = calculate_inter_cluster_distance(centroids[i], centroids[j])
                if intra_distances[i] + intra_distances[j] > 0:
                    ratio = (intra_distances[i] + intra_distances[j]) / inter_distance
                    max_ratio = max(max_ratio, ratio)
        db_index += max_ratio
    
    db_index /= n_clusters
    
    return db_index



def calculate_pairwise_distance(point1, point2, metric='euclidean'):
    if metric == 'euclidean':
        return np.linalg.norm(point1 - point2)
    elif metric == 'manhattan':
        return np.sum(np.abs(point1 - point2))
    elif metric == 'cosine':
        dot_product = np.dot(point1, point2)
        norm_product = np.linalg.norm(point1) * np.linalg.norm(point2)
        return 1 - (dot_product / (norm_product + 1e-10))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def calculate_metric_distance(point, centroid, metric):
    if metric == 'euclidean':
        return np.linalg.norm(point - centroid)
    elif metric == 'cosine':
        dot_product = np.dot(point, centroid)
        norm_product = np.linalg.norm(point) * np.linalg.norm(centroid)
        return 1 - (dot_product / (norm_product + 1e-10))
    elif metric == 'manhattan':
        return np.sum(np.abs(point - centroid))





def generate_data(distribution_type, n_samples, n_dimensions):
    if distribution_type == "Gaussian":
        return make_blobs(n_samples=n_samples, n_features=n_dimensions, centers=1)[0] 

    elif distribution_type == "Ring":
        if n_dimensions < 2:
            raise ValueError("Ring distribution requires at least 2 dimensions.")
        return make_circles(n_samples=n_samples, factor=0.5, noise=0.05)[0]

    elif distribution_type == "Spiral":
        if n_dimensions < 2:
            raise ValueError("Spiral distribution requires at least 2 dimensions.")
        theta = np.linspace(0, 4 * np.pi, n_samples)
        r = theta
        x = r * np.cos(theta) + np.random.normal(scale=0.1, size=n_samples)
        y = r * np.sin(theta) + np.random.normal(scale=0.1, size=n_samples)
        points = np.column_stack((x, y))
        if n_dimensions > 2:
            additional_dims = np.random.normal(size=(n_samples, n_dimensions - 2))
            points = np.column_stack((points, additional_dims))
        return points

    elif distribution_type == "Moon":
        if n_dimensions != 2:
            raise ValueError("Moon distribution is only defined in 2D.")
        return make_moons(n_samples=n_samples, noise=0.1)[0]

    elif distribution_type == "Uniform":
        return np.random.uniform(-5, 5, (n_samples, n_dimensions))

    elif distribution_type == "Swiss Roll":
        if n_dimensions != 3:
            raise ValueError("Swiss Roll is only defined in 3D.")
        return make_swiss_roll(n_samples=n_samples, noise=0.1)[0]
    elif distribution_type == "Custom":
        if n_dimensions != 2:
            raise ValueError("Moon distribution is only defined in 2D.")
        x = np.concatenate((np.random.normal(loc=-1, scale=0.1, size=n_samples // 4),
                            np.random.normal(loc=1, scale=0.1, size=n_samples // 4),
                            np.random.normal(loc=0, scale=0.1, size=n_samples // 4),
                            np.random.normal(loc=0, scale=0.1, size=n_samples // 4)))
        y = np.concatenate((np.random.normal(loc=-1, scale=0.1, size=n_samples // 4),
                            np.random.normal(loc=1, scale=0.1, size=n_samples // 4),
                            np.random.normal(loc=2, scale=0.1, size=n_samples // 4),
                            np.random.normal(loc=0, scale=0.1, size=n_samples // 4)))
        return np.column_stack((x, y))
    else:
        raise ValueError("Invalid distribution type.")