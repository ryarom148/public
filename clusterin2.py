mport pandas as pd
import numpy as np
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from saxpy.sax import sax_via_window
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Step 1: Convert list of dictionaries to pandas DataFrame
def create_dataframe(data):
    df = pd.DataFrame(data)
    df['label'] = df['client_id'] + '_' + df['month']
    df = df.set_index('label')
    return df

# Step 2: K-Shape clustering
def kshape_clustering(df, max_clusters=10):
    # Prepare data for K-Shape
    X = np.array(df['timeseries'].tolist())
    X = TimeSeriesScalerMeanVariance().fit_transform(X)

    # Find optimal number of clusters
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kshape = KShape(n_clusters=n_clusters, random_state=42)
        cluster_labels = kshape.fit_predict(X)
        score = silhouette_score(X.reshape(X.shape[0], -1), cluster_labels)
        silhouette_scores.append(score)

    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

    # Perform K-Shape clustering with optimal number of clusters
    kshape = KShape(n_clusters=optimal_clusters, random_state=42)
    df['kshape_cluster'] = kshape.fit_predict(X)

    return df, optimal_clusters

# Step 3: SAX clustering
def sax_clustering(df, word_size=8, alphabet_size=4, n_clusters=5):
    # Convert time series to SAX representation
    sax_representations = []
    for ts in df['timeseries']:
        sax_rep = sax_via_window(ts, word_size, alphabet_size, window_size=len(ts))
        sax_representations.append(''.join(sax_rep))
    
    df['sax_representation'] = sax_representations

    # Perform clustering on SAX representations
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['sax_cluster'] = kmeans.fit_predict(pd.get_dummies(df['sax_representation'].apply(list)))

    return df

# Main function to run the entire process
def process_timeseries_data(data):
    # Create DataFrame
    df = create_dataframe(data)

    # Perform K-Shape clustering
    df, optimal_kshape_clusters = kshape_clustering(df)

    # Perform SAX clustering
    df = sax_clustering(df)

    return df, optimal_kshape_clusters

# Example usage
data = [
    {'client_id': 'A', 'month': 'Jan', 'timeseries': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]},
    {'client_id': 'B', 'month': 'Jan', 'timeseries': [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]},
    # Add more data points here
]

result_df, optimal_clusters = process_timeseries_data(data)
print(f"Optimal number of clusters for K-Shape: {optimal_clusters}")
print(result_df)
