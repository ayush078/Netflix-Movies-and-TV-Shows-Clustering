
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the featured dataset
def load_featured_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Featured dataset loaded successfully from {file_path}. Shape:", df.shape)
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# Scale the data
def scale_data(df):
    # Ensure all columns are numeric
    numeric_df = df.select_dtypes(include=[np.number])
    print(f"Selected {numeric_df.shape[1]} numeric columns out of {df.shape[1]} total columns")
    
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(numeric_df)
    return scaled_df, numeric_df.columns

# Apply PCA for dimensionality reduction
def apply_pca(scaled_df, n_components=0.95):
    pca = PCA(n_components=n_components) # Retain 95% of variance
    pca_result = pca.fit_transform(scaled_df)
    print(f"\nPCA applied. Original dimensions: {scaled_df.shape[1]}, Reduced dimensions: {pca_result.shape[1]}")
    return pca_result

# K-Means Clustering
def kmeans_clustering(scaled_df):
    # Determine the optimal number of clusters using the elbow method
    inertia = []
    for n in range(2, 11):
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        kmeans.fit(scaled_df)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 11), inertia, marker=".")
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.savefig("kmeans_data.png")  
    plt.close()

    # Apply K-Means with the optimal number of clusters (let's assume 5 for now)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(scaled_df)
    return kmeans_labels

# Hierarchical Clustering
def hierarchical_clustering(scaled_df):
    # Using a subset of data for hierarchical clustering due to computational intensity
    # No need to subset if PCA is applied to the full dataset
    hierarchical = AgglomerativeClustering(n_clusters=5)
    hierarchical_labels = hierarchical.fit_predict(scaled_df)
    return hierarchical_labels

# DBSCAN Clustering
def dbscan_clustering(scaled_df):
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(scaled_df)
    return dbscan_labels

# Evaluate clusters
def evaluate_clusters(scaled_df, labels, algorithm_name):
    if len(set(labels)) > 1:
        silhouette = silhouette_score(scaled_df, labels)
        davies_bouldin = davies_bouldin_score(scaled_df, labels)
        print(f"\n{algorithm_name} Evaluation:")
        print(f"Silhouette Score: {silhouette}")
        print(f"Davies-Bouldin Index: {davies_bouldin}")
    else:
        print(f"\n{algorithm_name} resulted in a single cluster or noise. Evaluation metrics are not applicable.")

# Visualize clusters with PCA
def visualize_clusters_pca(scaled_df, labels, algorithm_name):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_df)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=labels, palette="viridis", s=50)
    plt.title(f"Clusters from {algorithm_name} (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Cluster")
    plt.savefig(f"{algorithm_name}_clusters_pca.png")  
    plt.close()

# Main execution
if __name__ == "__main__":
    file_path = "netflix_features.csv" 
    netflix_features = load_featured_data(file_path)
    
    if netflix_features is not None:
        scaled_features, feature_columns = scale_data(netflix_features)
        print(f"Using features: {list(feature_columns)}")
        
        pca_features = apply_pca(scaled_features) # Apply PCA here

        # K-Means
        kmeans_labels = kmeans_clustering(pca_features)
        evaluate_clusters(pca_features, kmeans_labels, "K-Means")
        visualize_clusters_pca(pca_features, kmeans_labels, "K-Means")

        # Hierarchical Clustering
        hierarchical_labels = hierarchical_clustering(pca_features)
        evaluate_clusters(pca_features, hierarchical_labels, "Hierarchical")
        visualize_clusters_pca(pca_features, hierarchical_labels, "Hierarchical")

        # DBSCAN
        dbscan_labels = dbscan_clustering(pca_features)
        evaluate_clusters(pca_features, dbscan_labels, "DBSCAN")
        visualize_clusters_pca(pca_features, dbscan_labels, "DBSCAN")

        print("\nClustering analysis complete. Visualizations saved in the current directory.")
    else:
        print("Failed to load featured data. Please ensure netflix_features.csv exists in the current directory.")


