import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score, adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure
import warnings

class KMeansClustering:
    def __init__(self,data):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        self.data = data
        self.X = pd.DataFrame(data=self.data.data, columns=self.data.feature_names)
        self.Y = pd.DataFrame(data=self.data.target, columns=['target'])['target']

    def calculate_wcss(self, points, kmax):
        sse = []
        for k in range(1, kmax + 1):
            kmeans = KMeans(n_clusters=k).fit(points)
            sse.append(kmeans.inertia_)
        return sse

    def plot_sse(self, kmax=10):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, kmax + 1), self.calculate_wcss(self.X, kmax), 'g*-')
        plt.xlabel('Number of clusters')
        plt.ylabel('SSE Score')
        plt.title('SSE Method For Optimal k')
        plt.show()

    def kmeans_clustering(self, n_clusters=3):
        kmeans = KMeans(n_clusters=n_clusters).fit(self.X)
        return kmeans

    def plot_clusters_pca(self, kmeans):
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.X)
        plt.figure(figsize=(10, 6))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans.labels_)
        plt.title('KMeans Clustering (PCA reduced)')
        plt.show()

    def internal_measures(self, kmeans):
        silhouette = silhouette_score(self.X, kmeans.labels_)
        calinski_harabasz = calinski_harabasz_score(self.X, kmeans.labels_)
        davies_bouldin = davies_bouldin_score(self.X, kmeans.labels_)
        return silhouette, calinski_harabasz, davies_bouldin

    def external_measures(self, kmeans):
        adjusted_rand = adjusted_rand_score(np.array(self.Y), kmeans.labels_)
        normalized_mutual_info = normalized_mutual_info_score(np.array(self.Y), kmeans.labels_)
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(np.array(self.Y), kmeans.labels_)
        return adjusted_rand, normalized_mutual_info, homogeneity, completeness, v_measure



kmeans_clustering = KMeansClustering(load_iris())
kmeans = kmeans_clustering.kmeans_clustering()
kmeans_clustering.plot_sse()
kmeans_clustering.plot_clusters_pca(kmeans)
print("Silhouette Score, Calinski Harabasz Score, Davies Bouldin Score:", kmeans_clustering.internal_measures(kmeans))
print("Adjusted Rand Score, Normalized Mutual Info Score, Homogeneity, Completeness, V Measure:", kmeans_clustering.external_measures(kmeans))
