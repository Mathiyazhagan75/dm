import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import Birch, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure
from sklearn.model_selection import ParameterGrid
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from HiPart.clustering import DePDDP
from HiPart.visualizations import dendrogram_visualization
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import warnings
warnings.simplefilter(action='ignore')

class ClusteringAnalysis:
    def __init__(self, data, labels):
        self.X = data
        self.data = data
        self.Y = labels
        self.X = self.preprocess()
    
    def preprocess(self):
        self.X = pd.DataFrame(StandardScaler().fit_transform(self.X), index=self.X.index, columns=self.X.columns)
        return self.X
        
    def plot_silhouette_scores(self, model, modelname):
        silhouette_scores = []
        cluster_sizes = range(2, 8)
        for n_cluster in cluster_sizes:
            if modelname == "DeDDP":
                model_instance = DePDDP(max_clusters_number=n_cluster).fit(self.data.values)
            else:
                model_instance = model(n_clusters=n_cluster)
                labels = model_instance.fit_predict(self.X)
            # print(silhouette_score(self.X, labels))
            silhouette_scores.append(silhouette_score(self.X, model_instance.labels_))
        # print(silhouette_scores)
        plt.figure(figsize=(21, 7))
        plt.bar(range(2, 8), silhouette_scores) 
        plt.title(f'{modelname}: Number of Clusters vs. Silhouette Score', fontsize=10)
        plt.xlabel('Number of Clusters', fontsize=20) 
        plt.ylabel('Silhouette Score', fontsize=20) 
        plt.show()
        max_score_index = np.argmax(silhouette_scores)
        return cluster_sizes[max_score_index]
            
    def evaluate_clustering(self,labels):
        scores = {}
        scores['Silhouette Score'] = silhouette_score(self.X, labels)
        scores['Calinski-Harabasz Score'] = calinski_harabasz_score(self.X,labels)
        scores['Davies-Bouldin Score'] = davies_bouldin_score(self.X, labels)
        scores['Adjusted Rand Index'] = adjusted_rand_score(np.array(self.Y), labels)
        scores['Normalized Mutual Information'] = normalized_mutual_info_score(np.array(self.Y), labels)
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(np.array(self.Y), labels)
        scores['Homogeneity'] = homogeneity
        scores['Completeness'] = completeness
        scores['V-Measure'] = v_measure
        return scores

class BirchClustering(ClusteringAnalysis):
    def __init__(self, data, branching_factor, threshold,labels):
        super().__init__(data, labels)
        self.branching_factor = branching_factor
        self.threshold = threshold
        self.model = None
    
    def find_optimal_params(self):
        param_grid = {
            'branching_factor': [50, 100, 200, 300],
            'threshold': [0.1, 0.5, 1.0, 1.5]
        }
        best_score = -1
        best_params = None
        for params in ParameterGrid(param_grid):
            birch = Birch(**params).fit(self.X)
            labels = birch.predict(self.X)
            score = silhouette_score(self.X, labels)
            if score > best_score:
                best_score = score
                best_params = params
        self.branching_factor = best_params['branching_factor']
        self.threshold = best_params['threshold']

    def perform_clustering(self):
        self.find_optimal_params()
        n = self.plot_silhouette_scores(Birch,"BIRCH")
        print("Hey")
        print(self.branching_factor, self.threshold,n)
        birch = Birch(branching_factor=self.branching_factor, threshold=self.threshold,n_clusters = n).fit(self.X)
        self.labels = birch.predict(self.X)
        pca_result = PCA(n_components=2).fit_transform(self.data)
        plt.figure(figsize=(10, 6))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=birch.labels_)
        plt.title('BIRCH Clustering (PCA reduced)')
        plt.show()
        print("Evaluation Scores:")
        print(self.evaluate_clustering(birch.labels_))
    
    def plot_silhouette(self):
        cluster_size = self.plot_silhouette_scores(Birch, "BIRCH")
        print(cluster_size)

class AgglomerativeClusteringAnalysis(ClusteringAnalysis):
    def __init__(self, data, n_clusters,labels):
        super().__init__(data, labels)
        self.n_clusters = n_clusters
        self.model = None
        
    def perform_clustering(self):
        n = self.plot_silhouette_scores(AgglomerativeClustering, "Agglomerative")
        print(n)
        agg = AgglomerativeClustering(n_clusters=n).fit(self.X)
        self.labels = agg.labels_
        pca_result = PCA(n_components=2).fit_transform(self.data)
        self.model = agg
        plt.figure(figsize=(10, 6))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=agg.labels_)
        plt.title('Agglomerative Clustering (PCA reduced)')
        plt.show()
        print("Evaluation Scores:")
        print(self.evaluate_clustering(agg.labels_))
        
    def plot_silhouette(self):
         self.plot_silhouette_scores(AgglomerativeClustering, "Agglomerative")
    def plot_dendrogram(self):
        linkage_matrix = linkage(self.X, method='ward')
        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()

class DivisiveClusteringAnalysis(ClusteringAnalysis):
    def __init__(self, data, n_clusters,labels):
        super().__init__(data, labels)
        self.n_clusters = n_clusters
        self.model = None
        
    def perform_clustering(self):
        n = self.plot_silhouette_scores(DePDDP, "DeDDP")
        print(n)
        depddp = DePDDP(max_clusters_number=n).fit(self.X.values)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X)
        plt.figure(figsize=(10,6))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=depddp.labels_)
        plt.title('Divisive Clustering (PCA reduced)')
        plt.show()
        plt.figure(figsize=(10, 7))
        plt.title("Dendrogram")
        dendrogram_visualization(depddp)
        plt.show()
        print("Evaluation Scores:")
        print(self.evaluate_clustering(depddp.labels_))
    def plot_silhouette(self):
         self.plot_silhouette_scores(AgglomerativeClustering, "Agglomerative")
   
class DBSCANClusteringAnalysis(ClusteringAnalysis):
    def __init__(self, data, eps, labels,min_samples=None):
        super().__init__(data, labels)
        self.min_samples = None
        self.eps = eps
        self.model = None
        
    def perform_clustering(self):
        # Compute k-distance graph
        MIN_PTS = 2 * self.X.shape[1]
        self.min_samples = MIN_PTS
        nearest_neighbors = NearestNeighbors(n_neighbors=MIN_PTS)
        neighbors = nearest_neighbors.fit(self.X)
        distances, indices = neighbors.kneighbors(self.X)
        distances = np.sort(distances[:, MIN_PTS-1], axis=0)
        
        # Plot the k-distance graph
        plt.figure(figsize=(5, 5))
        plt.plot(distances)
        plt.xlabel("Points sorted according to distance of "+str(MIN_PTS)+"th nearest neighbor")
        plt.ylabel(str(MIN_PTS)+"th nearest neighbor distance")
        plt.show()
        
        # Perform DBSCAN clustering
        dbscan = DBSCAN(min_samples=self.min_samples, eps=self.eps).fit(self.X)
        self.labels = dbscan.labels_
        core_samples_mask = np.zeros_like(self.labels, dtype=bool)
        core_samples_mask[dbscan.core_sample_indices_] = True
        
        # Plot DBSCAN clustering result
        self.plot_dbscan(self.X, self.labels, core_samples_mask, len(set(self.labels)) - (1 if -1 in self.labels else 0))
        
        print("Evaluation Scores:")
        print("Silhouette_score: ", silhouette_score(self.X, self.labels))
        print("Calinski_Harabasz_score", calinski_harabasz_score(self.X, self.labels))
        print("Davies_Bouldin_score", davies_bouldin_score(self.X, self.labels))
        print("Adjusted_Rand_score", adjusted_rand_score(np.array(self.Y), self.labels))
        print("Normalized_Mutual_Info_score", normalized_mutual_info_score(np.array(self.Y), self.labels))
        print("Homogeneity_Completeness_V_Measure", homogeneity_completeness_v_measure(np.array(self.Y), self.labels))
    
    def plot_dbscan(self, X, labels, core_samples_mask, n_clusters_):
        unique_labels = set(labels)
        pca_X = PCA(n_components=2).fit_transform(X)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

        for cluster_id, color in zip(unique_labels, colors):
            # Use black for noise (label -1)
            if cluster_id == -1:
                color = [0, 0, 0, 1]

            cluster_mask = labels == cluster_id

            core_samples_in_cluster = pca_X[cluster_mask & core_samples_mask]
            plt.plot(core_samples_in_cluster[:, 0], core_samples_in_cluster[:, 1], 'o', 
                     markerfacecolor=tuple(color), markeredgecolor='k', markersize=14)
            non_core_samples_in_cluster = pca_X[cluster_mask & ~core_samples_mask]
            plt.plot(non_core_samples_in_cluster[:, 0], non_core_samples_in_cluster[:, 1], 'o', 
                     markerfacecolor=tuple(color), markeredgecolor='k', markersize=6)

        plt.title(f"Estimated number of clusters: {n_clusters_}")
        plt.show()

# # Load Iris dataset
# iris = load_iris()
# X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# Y = pd.DataFrame(iris.target, columns=['target'])['target']

# # Instantiate BirchClustering
# birch_cluster = BirchClustering(X, branching_factor=50, threshold=1.5, labels = Y)

# # Perform Birch clustering
# birch_cluster.perform_clustering()

# # Instantiate AgglomerativeClusteringAnalysis
# agg_cluster = AgglomerativeClusteringAnalysis(X, n_clusters=3,labels = Y)

# # Perform Agglomerative clustering
# agg_cluster.perform_clustering()
# #
# # Plot dendrogram for Agglomerative clustering
# agg_cluster.plot_dendrogram()

# div_cluster = DivisiveClusteringAnalysis(X, n_clusters=3,labels = Y)
# div_cluster.perform_clustering()
# db = DBSCANClusteringAnalysis(X,eps = 0.875,labels = Y)
# db.perform_clustering()


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score, adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure
# import warnings

class KMeansClustering:
    def __init__(self,data):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        self.data = data
        self.X = pd.DataFrame(data=self.data.data, columns=self.data.feature_names)
        self.X = self.preprocess(self.X)
        self.Y = pd.DataFrame(data=self.data.target, columns=['target'])['target']
    
    def preprocess(self,df):
        df = df.copy(deep=True)
        return pd.DataFrame(StandardScaler().fit_transform(df), index=df.index, columns=df.columns)

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


# # Example Usage:
# kmeans_clustering = KMeansClustering(load_iris())
# kmeans = kmeans_clustering.kmeans_clustering()
# kmeans_clustering.plot_sse()
# kmeans_clustering.plot_clusters_pca(kmeans)
# print("Silhouette Score, Calinski Harabasz Score, Davies Bouldin Score:", kmeans_clustering.internal_measures(kmeans))
# print("Adjusted Rand Score, Normalized Mutual Info Score, Homogeneity, Completeness, V Measure:", kmeans_clustering.external_measures(kmeans))

