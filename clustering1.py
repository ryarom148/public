import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.utils import to_time_series_dataset
from sklearn.mixture import BayesianGaussianMixture
import matplotlib.pyplot as plt

def create_dataframe(data):
    df = pd.DataFrame(data)
    df['label'] = df['client_id'] + '_' + df['month']
    df = df.set_index('label')
    return df

def prepare_data(df):
    X = np.array(df['timeseries'].tolist())
    X = TimeSeriesScalerMeanVariance().fit_transform(X)
    return X

def hierarchical_clustering(X, n_clusters=None, max_clusters=10):
    best_score = -1
    best_n_clusters = 2
    scores = []
    
    for n in range(2, min(len(X), max_clusters + 1)):
        model = TimeSeriesKMeans(n_clusters=n, metric="dtw", max_iter=10)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels, metric="dtw")
        scores.append(score)
        if score > best_score:
            best_score = score
            best_n_clusters = n
    
    final_model = TimeSeriesKMeans(n_clusters=best_n_clusters, metric="dtw", max_iter=10)
    labels = final_model.fit_predict(X)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, min(len(X), max_clusters + 1)), scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Hierarchical Clustering: Silhouette Score vs Number of Clusters')
    plt.show()
    
    return labels, best_n_clusters, final_model

def dba_clustering(X, n_clusters=None, max_clusters=10):
    best_score = -1
    best_n_clusters = 2
    scores = []
    
    for n in range(2, min(len(X), max_clusters + 1)):
        model = TimeSeriesKMeans(n_clusters=n, metric="dtw", max_iter=10, average_method="dba")
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels, metric="dtw")
        scores.append(score)
        if score > best_score:
            best_score = score
            best_n_clusters = n
    
    final_model = TimeSeriesKMeans(n_clusters=best_n_clusters, metric="dtw", max_iter=10, average_method="dba")
    labels = final_model.fit_predict(X)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, min(len(X), max_clusters + 1)), scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('DBA Clustering: Silhouette Score vs Number of Clusters')
    plt.show()
    
    return labels, best_n_clusters, final_model

def dpgmm_clustering(X, max_clusters=10):
    X_flat = X.reshape(X.shape[0], -1)
    dpgmm = BayesianGaussianMixture(n_components=max_clusters, covariance_type='full', weight_concentration_prior=1e-2,
                                    weight_concentration_prior_type='dirichlet_process', init_params='random')
    dpgmm.fit(X_flat)
    labels = dpgmm.predict(X_flat)
    n_clusters = len(np.unique(labels))
    return labels, n_clusters, dpgmm

class TimeSeriesClusterer:
    def __init__(self):
        self.hierarchical_model = None
        self.dba_model = None
        self.dpgmm_model = None
        self.scaler = TimeSeriesScalerMeanVariance()

    def fit(self, data):
        df = create_dataframe(data)
        X = self.scaler.fit_transform(np.array(df['timeseries'].tolist()))

        hierarchical_labels, hierarchical_n_clusters, self.hierarchical_model = hierarchical_clustering(X)
        dba_labels, dba_n_clusters, self.dba_model = dba_clustering(X)
        dpgmm_labels, dpgmm_n_clusters, self.dpgmm_model = dpgmm_clustering(X)

        df['hierarchical_cluster'] = hierarchical_labels
        df['dba_cluster'] = dba_labels
        df['dpgmm_cluster'] = dpgmm_labels

        print(f"Hierarchical Clustering: Found {hierarchical_n_clusters} clusters")
        print(f"DBA Clustering: Found {dba_n_clusters} clusters")
        print(f"DPGMM Clustering: Found {dpgmm_n_clusters} clusters")

        return df

    def predict(self, new_data):
        new_df = create_dataframe(new_data)
        X_new = self.scaler.transform(np.array(new_df['timeseries'].tolist()))

        hierarchical_pred = self.hierarchical_model.predict(X_new)
        dba_pred = self.dba_model.predict(X_new)
        dpgmm_pred = self.dpgmm_model.predict(X_new.reshape(X_new.shape[0], -1))

        new_df['hierarchical_cluster'] = hierarchical_pred
        new_df['dba_cluster'] = dba_pred
        new_df['dpgmm_cluster'] = dpgmm_pred

        return new_df

# Example usage
data = [
    {'client_id': 'A', 'month': 'Jan', 'timeseries': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]},
    {'client_id': 'B', 'month': 'Jan', 'timeseries': [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]},
    {'client_id': 'C', 'month': 'Feb', 'timeseries': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11]},
    {'client_id': 'D', 'month': 'Feb', 'timeseries': [5, 5, 5, 5, 10, 10, 10, 10, 15, 15, 15, 15, 20, 20, 20, 20, 25, 25, 25, 25, 30]},
    # Add more data points here
]

# Train the clusterer
clusterer = TimeSeriesClusterer()
result_df = clusterer.fit(data)
print("Training Results:")
print(result_df)

# Predict on new data
new_data = [
    {'client_id': 'E', 'month': 'Mar', 'timeseries': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42]},
    {'client_id': 'F', 'month': 'Mar', 'timeseries': [10, 10, 10, 10, 15, 15, 15, 15, 20, 20, 20, 20, 25, 25, 25, 25, 30, 30, 30, 30, 35]}
]

prediction_df = clusterer.predict(new_data)
print("\nPrediction Results:")
print(prediction_df)
