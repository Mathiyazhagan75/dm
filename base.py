
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
            model_instance = model(n_clusters=n_cluster)
            labels = model_instance.fit_predict(self.X)
            # print(silhouette_score(self.X, labels))
            silhouette_scores.append(silhouette_score(self.X, labels))
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

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
Y = pd.DataFrame(iris.target, columns=['target'])['target']

# Instantiate BirchClustering
birch_cluster = BirchClustering(X, branching_factor=50, threshold=1.5, labels = Y)

# Perform Birch clustering
birch_cluster.perform_clustering()

# Instantiate AgglomerativeClusteringAnalysis
agg_cluster = AgglomerativeClusteringAnalysis(X, n_clusters=3,labels = Y)

# Perform Agglomerative clustering
agg_cluster.perform_clustering()
#
# Plot dendrogram for Agglomerative clustering
agg_cluster.plot_dendrogram()


