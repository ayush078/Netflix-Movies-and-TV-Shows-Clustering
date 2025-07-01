import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("netflix_titles.csv")

# Fill missing values in 'description' column with empty string
df["description"] = df["description"].fillna("")

# Combine relevant text features for clustering
df["combined_features"] = df["title"] + " " + df["director"].fillna("") + " " + \
                            df["cast"].fillna("") + " " + df["listed_in"].fillna("") + " " + \
                            df["description"]

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(df["combined_features"])

# Determine optimal number of clusters using Elbow Method (for demonstration, we'll pick a number)
# In a real scenario, you'd iterate and plot inertia.
num_clusters = 5 # Example number of clusters

# K-Means Clustering
kmeans_model = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, random_state=42, n_init=10)
df["cluster"] = kmeans_model.fit_predict(tfidf_matrix)

# PCA for dimensionality reduction for visualization
pca = PCA(n_components=2)
principal_components = pca.fit_transform(tfidf_matrix.toarray())
df["pca_1"] = principal_components[:, 0]
df["pca_2"] = principal_components[:, 1]

# Visualize the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x="pca_1", y="pca_2", hue="cluster", data=df, palette="viridis", s=100, alpha=0.8)
plt.title("Netflix Content Clusters (PCA Reduced)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.savefig("netflix_clusters.png")
print("Clustering complete. Visualization saved as netflix_clusters.png")

# Display some titles from each cluster
for i in range(num_clusters):
    print(f"\nCluster {i} Titles:")
    print(df[df["cluster"] == i]["title"].head(5).tolist())

