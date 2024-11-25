import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted
import warnings
import dlib  # Correct Dlib import

warnings.filterwarnings("ignore")

def calculate_sigma_from_distances(distance_matrix, percentiles=[50], show_plot=False):
    """
    Calculate sigma for Gaussian kernel to convert distance to similarity.
    
    Parameters:
    - distance_matrix (np.ndarray): Distance matrix (n x n).
    - percentiles (list): Percentiles to calculate sigma, typically around 50th.
    - show_plot (bool): If True, plot sorted distances.
    
    Returns:
    - sigma (float): Calculated sigma for Gaussian kernel.
    """
    # Extract the upper triangle to avoid redundant values
    upper_tri_indices = np.triu_indices_from(distance_matrix, k=1)
    upper_tri_distances = distance_matrix[upper_tri_indices]
    
    # Sort distances
    sorted_distances = np.sort(upper_tri_distances)
    
    # Calculate sigma as the median or a chosen percentile
    sigma = np.percentile(sorted_distances, percentiles[0])
    
   
    return sigma


def distance_to_similarity_matrix(distance_matrix):
    """
    Convert a distance matrix to a similarity matrix using Gaussian kernel.
    
    Parameters:
    - distance_matrix (np.ndarray): Distance matrix (n x n).
    - sigma (float): Gaussian kernel sigma.
    - show_plot (bool): If True, display histogram of similarities.
    
    Returns:
    - similarity_matrix (np.ndarray): Converted similarity matrix.
    """
    sigma = calculate_sigma_from_distances(distance_matrix)
    
    similarity_matrix = np.exp(-np.square(distance_matrix) / (2 * sigma ** 2))
   

    return similarity_matrix
def build_edge_list(similarity_matrix, k=5):
    """
    Build a sorted edge list for Dlib's Chinese Whispers implementation.

    Parameters:
    - similarity_matrix (ndarray): A 2D square matrix of similarity scores.
    - k (int or None): The number of top neighbors to consider for each node.
                       If None, consider all neighbors.

    Returns:
    - edge_list (list of tuples): List of edges in the form (i, j, weight),
      where i < j and each edge is unique.
    """
    n_samples = similarity_matrix.shape[0]
    edge_dict = {}

    for i in range(n_samples):
        # Get indices of neighbors sorted by similarity descending
        neighbors = np.argsort(similarity_matrix[i])[::-1]
        neighbors = neighbors[neighbors != i]  # Exclude self-loops

        if k is not None:
            neighbors = neighbors[:k]

        for j in neighbors:
            a, b = sorted((i, j))  # Ensure a < b for undirected edges
            if (a, b) not in edge_dict:
                edge_dict[(a, b)] = similarity_matrix[a, b]
            # If (a, b) is already in edge_dict, do not add it again

    # Convert the dictionary to a sorted list of tuples
    edge_list = sorted(
        [(a, b, w) for (a, b), w in edge_dict.items()],
        key=lambda x: (x[0], x[1])
    )
    # Convert NumPy data types to native Python types
    converted_edge_list = [(int(a), int(b), float(w)) for a, b, w in edge_list]
    return converted_edge_list


import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted
import dlib  # Ensure dlib is properly installed and accessible

class CustomChineseWhispers(BaseEstimator, ClusterMixin):
    def __init__(self, iterations=20, k=5):
        """
        Custom Chinese Whispers clustering estimator using Dlib.

        Parameters:
        - iterations (int): Number of iterations to run the algorithm.
        - k (int): Number of top similar neighbors to consider for each node.
                   If set to None, all neighbors are considered.
        """
        self.iterations = iterations
        self.k = k

    def fit(self, distance_matrix, y=None):
        """
        Fit the Chinese Whispers model using the precomputed distance matrix.

        Parameters:
        - distance_matrix (ndarray): Precomputed distance matrix of shape (n_samples, n_samples).
        - y: Ignored.

        Returns:
        - self
        """
        # Check if the distance matrix is square
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("Distance matrix must be square.")

        self.distance_matrix_ = distance_matrix
        #n_samples = self.distance_matrix_.shape[0]

        # Convert the distance matrix to a similarity matrix
        similarity_matrix_ = distance_to_similarity_matrix(self.distance_matrix_)

        # Build edge list for Dlib
        edge_list = build_edge_list(similarity_matrix_, self.k)

        # Convert edge list to Dlib's required format
        #dlib_edges = [dlib.sample_pair(edge[0], edge[1], edge[2]) for edge in edge_list]

        # Initialize labels list
        labels = []

        # Perform Chinese Whispers clustering using Dlib
        #num_clusters = dlib.chinese_whispers(dlib_edges, labels, self.iterations)
        # Apply Chinese Whispers algorithm using Dlib's implementation
        labels = dlib.chinese_whispers(edge_list)

        # Assign labels to the class
        self.labels_ = np.array(labels)

        # Determine cluster centers and other metrics
        unique_labels = set(self.labels_)
        self.set_cluster_centers_indices(unique_labels)
        self.set_outlier_thresholds(unique_labels)

        # Compute the overall outlier ratio
        self.outlier_ratio_ = np.sum(self.labels_ == -1) / len(self.labels_)
        return self

   

    def set_cluster_centers_indices(self, unique_labels):
        """
        Determine and store cluster center indices.

        Parameters:
        - unique_labels (set): Set of unique cluster labels.
        """
        self._cluster_centers_indices_ = {}
        for label in unique_labels:
            cluster_indices = np.where(self.labels_ == label)[0]
            if len(cluster_indices) == 0:
                continue
            # Compute the medoid: point with minimum total distance to others in the cluster
            sub_distance_matrix = self.distance_matrix_[np.ix_(cluster_indices, cluster_indices)]
            total_distances = sub_distance_matrix.sum(axis=1)
            medoid_idx = cluster_indices[np.argmin(total_distances)]  # Minimum total distance
            self._cluster_centers_indices_[label] = medoid_idx

    def set_outlier_thresholds(self, unique_labels):
        """
        Determine and store outlier thresholds for each cluster.

        Parameters:
        - unique_labels (set): Set of unique cluster labels.
        """
        self.outlier_thresholds_ = {}
        for label in unique_labels:
            medoid_idx = self._cluster_centers_indices_[label]
            cluster_mask = self.labels_ == label
            distances_to_medoid = self.distance_matrix_[cluster_mask, medoid_idx]
            # Define the threshold as the 95th percentile of distances
            threshold = np.percentile(distances_to_medoid, 95)
            self.outlier_thresholds_[label] = threshold

    def _compute_outlier_mask(self):
        """
        Compute a boolean mask indicating which samples are outliers based on thresholds.

        Returns:
        - outlier_mask (ndarray): Boolean array where True indicates an outlier.
        """
        outlier_mask = np.zeros(len(self.labels_), dtype=bool)
        for label, medoid_idx in self._cluster_centers_indices_.items():
            cluster_mask = self.labels_ == label
            distances_to_medoid = self.distance_matrix_[cluster_mask, medoid_idx]
            threshold = self.outlier_thresholds_.get(label, np.inf)
            outliers = distances_to_medoid > threshold
            outlier_mask[cluster_mask] = outliers
        return outlier_mask

    def predict(self, X, check_outliers=True):
        """
        Predict cluster labels for new data based on the distance to cluster centers.

        Parameters:
        - X (ndarray): Distance matrix of shape (n_samples_new, n_clusters).
                       Each column corresponds to a cluster center.
        - check_outliers (bool): Whether to check for outliers based on thresholds.

        Returns:
        - labels (ndarray): Cluster labels for each sample. Outliers are marked as -1.
        """
        check_is_fitted(self, ['cluster_centers_indices_'])

        # Assign each sample to the nearest cluster center
        labels = np.argmin(X, axis=1)

        if check_outliers:
            # Retrieve the minimal distance for each sample
            min_distances = X[np.arange(X.shape[0]), labels]
            # Retrieve corresponding cluster labels
            cluster_labels = list(self._cluster_centers_indices_.keys())
            assigned_cluster_labels = [cluster_labels[label] for label in labels]
            # Retrieve thresholds for assigned clusters
            thresholds = np.array([self.outlier_thresholds_.get(label, np.inf) for label in assigned_cluster_labels])
            # Identify outliers
            outliers = min_distances > thresholds
            labels[outliers] = -1

        return labels
    

   

    @property
    def cluster_centers_indices_(self):
        """
        Get the indices of cluster centers (medoids).

        Returns:
        - centers (list): List of medoid indices for each cluster.
        """
        if not hasattr(self, '_cluster_centers_indices_'):
            raise AttributeError("The cluster_centers_indices_ attribute is not set.")
        return list(self._cluster_centers_indices_.values())

    @cluster_centers_indices_.setter
    def cluster_centers_indices_(self, value):
        self._cluster_centers_indices_ = value

    def fit_predict(self, X, y=None):
        """
        Fit the model and predict the cluster labels for the training data.

        Parameters:
        - X (ndarray): Precomputed similarity matrix of shape (n_samples, n_samples).
        - y: Ignored.

        Returns:
        - labels (ndarray): Cluster labels for each sample.
        """
        self.fit(X)
        # Apply outlier thresholds to training data
        outlier_mask = self._compute_outlier_mask()
        labels = self.labels_.copy()
        labels[outlier_mask] = -1
        self.outlier_ratio_ = np.sum(labels == -1) / len(labels)
        self.labels_ = labels
        return self.labels_

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters:
        - deep (bool): If True, will return the parameters for this estimator and
                       contained subobjects that are estimators.

        Returns:
        - params (dict): Parameter names mapped to their values.
        """
        return {'iterations': self.iterations, 'k': self.k}

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters:
        - **params: dict
            Estimator parameters.

        Returns:
        - self: Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self



# ########## Sample of use ################

# import numpy as np
# import matplotlib.pyplot as plt
# from tslearn.metrics import cdist_dtw
# from tslearn.datasets import CachedDatasets
# from sklearn.preprocessing import StandardScaler

# # Import the ChineseWhispersClustering class
# # Assume the class is defined in a file named chinese_whispers_clustering.py
# # from chinese_whispers_clustering import ChineseWhispersClustering

# # For this example, we'll define the class in the same script
# # Please ensure the class definition provided earlier is available in your environment

# # Generate synthetic time series data
# def generate_synthetic_data(n_samples=90, n_features=100, n_clusters=3, random_state=42):
#     np.random.seed(random_state)
#     X = []
#     labels = []
#     t = np.linspace(0, 4 * np.pi, n_features)
    
#     # Define distinct patterns for each cluster
#     patterns = []
#     for i in range(n_clusters):
#         phase_shift = i * np.pi / 2  # Phase shift to separate clusters
#         amplitude = 1 + i * 0.5      # Different amplitude for each cluster
#         frequency = 1 + i * 0.2      # Different frequency for each cluster
#         pattern = amplitude * np.sin(frequency * t + phase_shift)
#         patterns.append(pattern)
    
#     # Generate data for each cluster
#     samples_per_cluster = n_samples // n_clusters
#     for i in range(n_clusters):
#         pattern = patterns[i]
#         for _ in range(samples_per_cluster):
#             noise = np.random.normal(0, 0.2, n_features)  # Adjusted noise level
#             X.append(pattern + noise)
#             labels.append(i)
    
#     X = np.array(X)
#     labels = np.array(labels)
#     return X, labels

# # Main function demonstrating the use of ChineseWhispersClustering
# def main():
#     # Step 1: Generate synthetic time series data
#     X_train, labels_true = generate_synthetic_data(n_samples=90, n_features=100, n_clusters=3)
#     X_test, y_test_true = generate_synthetic_data(n_samples=30, n_features=100, n_clusters=3, random_state=24)

#     # for i in range(3):
#     #     plt.figure(figsize=(10, 2))
#     #     for j in range(3):  # Plot 3 samples from each cluster
#     #         index = i * (len(X_train) // 3) + j
#     #         plt.plot(X_train[index], label=f'Sample {index}, Cluster {labels_true[index]}')
#     #     plt.title(f'Cluster {i} Samples')
#     #     plt.legend()
#     #     plt.show()


#     # Optional: Standardize the data
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # Step 2: Compute the distance matrix using DTW
#     print("Computing distance matrix...")
#     distance_matrix = cdist_dtw(X_train_scaled)
    

#     # plt.figure(figsize=(10, 8))
#     # sns.heatmap(distance_matrix)
#     # plt.title("Distance Matrix Heatmap")
#     # plt.show()
#     # Step 3: Initialize and fit the ChineseWhispersClustering model
#     cw_clustering = CustomChineseWhispers(
#         k=3,  # Number of nearest neighbors; set k=-1 for full graph
#         iterations=10000,
        
#     )
#     print("Fitting the Chinese Whispers clustering model...")
#     cw_clustering.fit(distance_matrix)

#     # Step 4: Get the cluster labels
#     labels = cw_clustering.labels_
#     print("Cluster labels for training data:", labels)
#     from sklearn.metrics import adjusted_rand_score
#     ari_score = adjusted_rand_score(labels_true, labels)
#     print(f"Adjusted Rand Index (ARI): {ari_score:.2f}")
#     # Step 5: Plot the clusters (with dimensionality reduction)
#     # print("Plotting clusters...")
#     # cw_clustering.plot_clusters(X_train_scaled, reduce_dims=False)

#     # Step 6: Prepare test data for prediction
#     # Compute distance matrix between test samples and cluster centers
#     cluster_centers_indices = list(cw_clustering.cluster_centers_indices_.values())
#     X_centers = X_train_scaled[cluster_centers_indices]

#     def compute_distance_matrix(new_X, X_centers):
#         n_new = new_X.shape[0]
#         n_centers = X_centers.shape[0]
#         distance_matrix = np.zeros((n_new, n_centers))
#         for i in range(n_new):
#             for j in range(n_centers):
#                 # Compute DTW distance
#                 distance = cdist_dtw(new_X[i].reshape(1, -1), X_centers[j].reshape(1, -1))[0, 0]
#                 distance_matrix[i, j] = distance
#         return distance_matrix

#     print("Computing distance matrix for test data...")
#     new_distance_matrix = compute_distance_matrix(X_test_scaled, X_centers)

#     # Step 7: Predict clusters for new data
#     print("Predicting clusters for new data...")
#     new_labels = cw_clustering.predict(new_distance_matrix, outliers_check=True)
#     print("Predicted labels for test data:", new_labels)

#     # Step 8: Evaluate the clustering results (if true labels are known)
#     from sklearn.metrics import adjusted_rand_score
#     ari = adjusted_rand_score(y_test_true, new_labels)
#     print(f"Adjusted Rand Index for test data: {ari:.2f}")

#     # Step 9: Plot the test data clusters (with dimensionality reduction)
#     print("Plotting clusters for test data...")
#     plt.figure(figsize=(10, 7))
#     X_test_flat = X_test_scaled.reshape(X_test_scaled.shape[0], -1)
#     pca = PCA(n_components=2)
#     X_test_reduced = pca.fit_transform(X_test_flat)
#     scatter = plt.scatter(X_test_reduced[:, 0], X_test_reduced[:, 1], c=new_labels, cmap='tab20')
#     plt.title('Chinese Whispers Clustering - Test Data')
#     plt.xlabel('Component 1')
#     plt.ylabel('Component 2')
#     plt.legend(*scatter.legend_elements(), title="Clusters")
#     plt.show()

# if __name__ == "__main__":
#     main()