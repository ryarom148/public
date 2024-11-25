import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, inconsistent,maxinconsts
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.spatial.distance import squareform
from sklearn.utils.validation import check_is_fitted
from scipy.signal import savgol_filter
from kneed import KneeLocator
import matplotlib.pyplot as plt
import time

class CustomAgglomerativeClustering(BaseEstimator, ClusterMixin):
    def __init__(self, linkage_method="ward", optimization_method="double_derivative", depth=2):
        """
        Custom Agglomerative Clustering estimator.

        Parameters:
        - linkage_method (str): Linkage method to use ('ward', 'complete', 'average', etc.).
        - optimization_method (str): Method for finding the optimal number of clusters.
                                     Options: "double_derivative", "first_derivative", "inconsistency".
        - depth (int): Depth for inconsistency calculation when using the inconsistency method.
        """
        self.linkage_method = linkage_method
        if isinstance(optimization_method, dict):
                    self.optimization_method = optimization_method.get("method")
                    self.depth = optimization_method.get("depth")
        else:
                    self.optimization_method = optimization_method
        #self.optimization_method = optimization_method
                    self.depth = depth

    def fit(self, X, y=None):
        """
        Fit the Agglomerative Clustering model using the data or a precomputed distance matrix.

        Parameters:
        - X (ndarray): Data matrix of shape (n_samples, n_features) or precomputed distance matrix.
        - y: Ignored.

        Returns:
        - self
        """
        # Compute or load linkage matrix
        self.distance_matrix_ = X
        self.linkage_matrix_ = self._load_or_compute_linkage(X)
       
        # Determine the optimal number of clusters or cutoff distance
        if self.optimization_method == "double_derivative":
            cutoff_distance = self._optimum_clusters_elbow_double()
            criterion="distance"
        elif self.optimization_method == "first_derivative":
            cutoff_distance = self._optimum_clusters_elbow_first()
            criterion="distance"
        elif self.optimization_method == "inconsistency":
            cutoff_distance = self._optimum_clusters_inconsistency()
            criterion="inconsistent"
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")

        # Generate cluster labels using the cutoff distance
        self.labels_ = fcluster(self.linkage_matrix_, t=cutoff_distance, criterion=criterion)

        # Set cluster center indices
        unique_labels = set(self.labels_)
        self.set_cluster_centers_indices(unique_labels)

        # Set outlier thresholds
        self.set_outlier_thresholds(unique_labels)

        # Compute overall outlier ratio in the training data
        self.outlier_ratio_ = np.sum(self._compute_outlier_mask()) / len(self.labels_)

        return self

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
        check_is_fitted(self, ["_cluster_centers_indices_"])

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

    def set_cluster_centers_indices(self, unique_labels):
        """
        Determine and store cluster center indices.

        Parameters:
        - unique_labels (set): Set of unique cluster labels excluding noise.
        """
        cluster_centers_indices_ = {}
        for label in unique_labels:
            cluster_indices = np.where(self.labels_ == label)[0]
            if len(cluster_indices) == 0:
                continue
            # Extract the sub-distance matrix for the cluster
            sub_distance_matrix = self.distance_matrix_[np.ix_(cluster_indices, cluster_indices)]
            # Compute the sum of distances for each point in the cluster
            total_distances = sub_distance_matrix.sum(axis=1)
            # Identify the medoid (point with minimum total distance)
            medoid_idx = cluster_indices[np.argmin(total_distances)]
            cluster_centers_indices_[label] = medoid_idx
        self._cluster_centers_indices_ = cluster_centers_indices_

    def set_outlier_thresholds(self, unique_labels):
        """
        Determine and store outlier thresholds for each cluster.

        Parameters:
        - unique_labels (set): Set of unique cluster labels excluding noise.
        """
        self.outlier_thresholds_ = {}
        for label in unique_labels:
            medoid_idx = self._cluster_centers_indices_[label]
            cluster_mask = self.labels_ == label
            # Distances from points in the cluster to the medoid
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
            threshold = self.outlier_thresholds_[label]
            outliers = distances_to_medoid > threshold
            outlier_mask[cluster_mask] = outliers
        return outlier_mask

    def _load_or_compute_linkage(self, X):
        """
        Load the linkage matrix from a file or compute it from data.

        Parameters:
        - X (ndarray): Data matrix of shape (n_samples, n_features).

        Returns:
        - linkage_matrix (ndarray): The linkage matrix.
        """
        folder = "linkages"
        file_name = f"{self.linkage_method}.npy"
        file_path = os.path.join(folder, file_name)

        if os.path.exists(file_path):
            print(f"Loading linkage matrix from {file_path}")
            linkage_matrix = np.load(file_path)
        else:
            print(f"Computing linkage matrix using method: {self.linkage_method}")
            condensed_distance = squareform(distance_matrix)
            linkage_matrix = linkage(condensed_distance, method=self.linkage_method)
            # os.makedirs(folder, exist_ok=True)
            # np.save(file_path, linkage_matrix)

        return linkage_matrix

    def _optimum_clusters_elbow_double(self):
        """
        Determine the optimal number of clusters using the Double Derivative method.

        Returns:
        - cutoff_distance (float): Distance cutoff for clustering.
        """
        distances = self.linkage_matrix_[:, 2][:-1]
        x = np.arange(1, len(distances)+1)
        # Smooth the distances
        y = savgol_filter(distances, window_length=4, polyorder=2)
        knee_locator = KneeLocator(x, y, curve="convex",direction="increasing") #
        knee_value = knee_locator.knee_y
        optimal_index=knee_locator.knee
        knee_value=distances[optimal_index]
        #double_derivative = np.diff(distances, 2)
        #optimal_index = np.argmax(double_derivative) + 2  # Adjust for length difference
        print('#ofslusteres: ',len(distances) - optimal_index)
        return knee_value

    def _optimum_clusters_elbow_first(self):
        """
        Determine the optimal number of clusters using the First Derivative method.

        Returns:
        - cutoff_distance (float): Distance cutoff for clustering.
        """
        distances = self.linkage_matrix_[:, 2]
        first_derivative = np.diff(distances)
        optimal_index = np.argmax(first_derivative) + 1  # Adjust for length difference
        print('#ofslusteres: ',len(distances) - optimal_index)
        return distances[optimal_index]

    def _optimum_clusters_inconsistency(self):
        """
        Determine the optimal number of clusters using inconsistency statistics.

        Returns:
        - cutoff_distance (float): Distance cutoff for clustering.
        """
        inconsistency_stats = inconsistent(self.linkage_matrix_, self.depth)
        maxincons = maxinconsts(self.linkage_matrix_, inconsistency_stats)
        cutoff_distance = np.median(maxincons)
        #cutoff_distance = np.max(inconsistency_stats[:, -1])
        return cutoff_distance
    
    @property
    def cluster_centers_indices_(self):
        """
        Get the indices of cluster centers (medoids).

        Returns:
        - centers: dict
            Dictionary mapping cluster labels to medoid indices.
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
        - X (ndarray): Precomputed distance matrix of shape (n_samples, n_samples).
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

        Returns:
        - params (dict): Parameter names mapped to their values.
        """
        return {
            "linkage_method": self.linkage_method,
            "optimization_method": self.optimization_method,
            "depth": self.depth,
           
        }

    def set_params(self, **params):
        """
        Set parameters for this estimator.

        Parameters:
        - **params: dict
            Estimator parameters.

        Returns:
        - self: Estimator instance.
        """
        for key, value in params.items():
            if key == "optimization_method":
                # Handle cases where optimization_method is a dictionary
                if isinstance(value, dict):
                    self.optimization_method = value.get("optimization_method", self.optimization_method)
                    self.depth = value.get("depth", self.depth)
                else:
                    self.optimization_method = value
            else:
                setattr(self, key, value)
        return self
    
def compute_distance_accl(linkage_matrix):


        dist_sort, lchild, rchild = [], [], []
        for lnk in linkage_matrix:
            left_child = lnk[0]
            right_child = lnk[1]
            dist_left_right = lnk[2]
            lchild.append(left_child)
            rchild.append(right_child)
            dist_sort.append(dist_left_right)

        dist_sort = np.array(dist_sort)
        dist_sort_id = np.argsort(dist_sort)
       
        df_cv = pd.DataFrame()
        df_cv["level"] = np.arange(dist_sort.shape[0], 0, -1)
        df_cv["distance"] = dist_sort
        accl = df_cv["distance"].diff().diff()  # double derivative

        df_accl = pd.DataFrame()
        df_accl["level"] = np.arange(dist_sort.shape[0]+1, 1, -1)
        df_accl["accln"] = accl
        return df_cv, df_accl

from scipy.signal import savgol_filter
def optimum_cluster_elbow(linkage_matrix,minmax=False, plotloc=False):
        '''
        Gives the optimum number of clusters required to express the maximum difference in the similarity using the elbow method
        '''

        df_cv, df_accl = compute_distance_accl(linkage_matrix)
        x = np.arange(1, len(linkage_matrix[:,2]) )
        print(len(x))
        y = linkage_matrix[:,2][:-1]
        print(len(y))
        # Normalize the distances for consistent scaling across linkage methods
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
    
       

        # Smooth the distances
        y = savgol_filter(y, window_length=4, polyorder=2)

        # Adjust KneeLocator parameters based on distances being in increasing order
        knee_locator = KneeLocator(x, y, curve="convex", direction="increasing")
        #knee_locator = KneeLocator(x, y, curve="convex",direction="increasing") #
        knee_value = knee_locator.knee_y
        idx =knee_locator.knee+1 ## some how need the ajustments
        #idx = df_accl['accln'].argmax()
        opt_cluster = df_accl.loc[idx, 'level']
        df_cv_subset = df_cv[df_cv['level'] == opt_cluster]
        opt_distance = df_cv_subset['distance'].values[0]
        if plotloc:
            df_cv_subset2 = df_cv[df_cv['level'] == opt_cluster-1]
            opt_distance_next = df_cv_subset2['distance'].values[0]
            optdist_plotloc = opt_distance+(opt_distance_next-opt_distance)/2
            return opt_cluster, opt_distance, optdist_plotloc
        if minmax:
            mindist, maxdist = df_cv["distance"].min(), df_cv["distance"].max()
            return opt_cluster, opt_distance, (mindist, maxdist)

        return opt_cluster, opt_distance
kwargs_default = {
    'fontsize': 10,
    'figsize': (10, 8)
}
def plot_optimum_cluster(    linkage_matrix,
                             method,
                             max_d=None,
                             figname=None,
                             figsize=(10, 6),
                             plotpdf=True,
                             xlabel="Number of Clusters",
                             ylabel="Distance",
                             accl_label='Curvature',
                             accl_color="C10",
                             dist_label="DTW Distance",
                             dist_color="k",
                             xlim=40,
                             legend_outside=True,
                             fontsize=kwargs_default['fontsize'],
                             xlabelfont=kwargs_default['fontsize'],
                             ylabelfont=kwargs_default['fontsize']):
        '''
        :param xlim: x limits of the plot e.g., [1,2]
        :type xlim: list
        '''
        df_cv, df_accl = compute_distance_accl(linkage_matrix)
        tail_df = df_cv.tail(xlim).reset_index(drop=True)
        accl_df = df_accl.tail(xlim).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(tail_df["level"], tail_df["distance"],
                "o-", color=dist_color, label=dist_label)

        ax.plot(
            accl_df["level"],
            accl_df["accln"],
            "o-",
            color=accl_color,
            label=accl_label,
        )

        if max_d is None:
            max_d, opt_distance = optimum_cluster_elbow(linkage_matrix)

        ax.axvline(x=max_d, label="Optimum cluster", ls="--")

        ax.set_xlabel(xlabel, fontsize=xlabelfont)
        ax.set_ylabel(ylabel, fontsize=ylabelfont)
        plt.subplots_adjust(wspace=0.01)
        fig.align_xlabels()

        if legend_outside:
            plt.legend(fontsize=fontsize, bbox_to_anchor=(
                1.05, 1), loc='upper left')
        else:
            plt.legend(fontsize=fontsize)

        plt.xticks(np.arange(tail_df["level"].min()-1, tail_df["level"].max()+1, 1
                             ).astype(int), fontsize=xlabelfont)
        plt.yticks(fontsize=ylabelfont)
        # if xlim is not None:
        #     plt.xlim(xlim)
        plt.title(f"method: {method}")
        plt.show()
        # if figname:
        #     plt.savefig(figname, bbox_inches='tight')
        #     if plotpdf:
        #         figname_pdf = ".".join(figname.split(".")[:-1])+".pdf"
        #         ax.figure.savefig(figname_pdf,
        #                           bbox_inches='tight')
        #     plt.close()
        #     fig, ax = None, None
        # return fig, ax

def generate_agglo_param_grid(linkage_methods, optimization_methods,inconsistency_depths):
    # Create the parameter grid
    """
    Generate parameter grid for CustomAgglomerativeClustering in GridSearchCV-compatible format.

    Returns:
        dict: A dictionary with parameter names as keys and lists of parameter values as values.
    # """
    # # Define parameter values
    # linkage_methods = ['ward', 'complete', 'average', 'single']
    # optimization_methods = ['double_derivative', 'first_derivative']
    # inconsistency_depths = [2, 3, 4]

    # Prepare the grid in the required format
    parameter_grid = {
        'linkage_method': linkage_methods,
        'optimization_method': optimization_methods + [
            {'method': 'inconsistency', 'depth': depth} for depth in inconsistency_depths
        ]
    }
    return parameter_grid
def generate_linkages(distance_matrix, linkage_methods=['average', 'complete', 'single'],f_show=True): #,
    # Convert distance matrix to condensed form
    condensed_distance = squareform(distance_matrix)
    folder = "linkages"
        
    # Perform hierarchical clustering using 'average' linkage
    distance_thresholds =[]
    for method in linkage_methods:
        start_time = time.time()
        file_name = f"{method}.npy"
        file_path = os.path.join(folder, file_name)
        linkage_matrix = linkage(condensed_distance, method = method)
        elapsed_time = time.time() - start_time
        print(f"Linkage for Agglomerative methos {method} calculated in {elapsed_time:.2f} seconds")
        os.makedirs(folder, exist_ok=True)
        np.save(file_path, linkage_matrix)
        if f_show:
            plot_optimum_cluster(linkage_matrix,method)
    
    #return {'distance_threshold': distance_thresholds, 'linkage': linkage_methods}

