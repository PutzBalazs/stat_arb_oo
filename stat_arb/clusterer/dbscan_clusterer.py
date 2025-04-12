import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from typing import List
from ..core import Token
from .base_clusterer import Clusterer

class DBSCANClusterer(Clusterer):
    def __init__(self, tokens: List[Token], n_components: int = 2, eps: float = 0.5, min_samples: int = 2):
        super().__init__(tokens)
        self.n_components = n_components
        self.eps = eps
        self.min_samples = min_samples
        self.pca_features = None
        self.pca_model = None
        self.cluster_labels = None

    def _prepare_features(self) -> pd.DataFrame:
        """Prepare features for clustering using PCA"""
        # Create a matrix of log returns
        returns_matrix = pd.DataFrame()
        
        for token in self.tokens:
            if not token.log_returns.empty:
                returns_matrix[token.symbol] = token.log_returns
        
        # Drop any rows with NaN values
        returns_matrix = returns_matrix.dropna()
        
        if returns_matrix.empty:
            raise ValueError("No valid returns data available")
        
        # Perform PCA
        self.pca_model = PCA(n_components=min(self.n_components, len(returns_matrix.columns)))
        self.pca_features = self.pca_model.fit_transform(returns_matrix.T)  # Transpose to get tokens as rows
        
        # Create DataFrame with token symbols as index
        features_df = pd.DataFrame(
            self.pca_features,
            index=[token.symbol for token in self.tokens if not token.log_returns.empty],
            columns=[f'PC{i+1}' for i in range(self.pca_features.shape[1])]
        )
        
        return features_df

    def run(self) -> List[List[Token]]:
        """Run DBSCAN clustering on PCA features"""
        # Prepare features
        features_df = self._prepare_features()
        
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.cluster_labels = dbscan.fit_predict(features_df)
        
        # Create clusters
        self.clusters = []
        unique_labels = set(self.cluster_labels)
        
        for label in unique_labels:
            if label != -1:  # -1 is noise in DBSCAN
                cluster_indices = np.where(self.cluster_labels == label)[0]
                cluster_tokens = [self.tokens[i] for i in cluster_indices]
                self.clusters.append(cluster_tokens)
        
        return self.clusters

    def visualize(self):
        """Visualize the clustering results"""
        if self.pca_features is None or self.cluster_labels is None:
            raise ValueError("Run clustering first")
        
        plt.figure(figsize=(10, 8))
        
        # Plot clusters
        for label in set(self.cluster_labels):
            if label == -1:
                # Plot noise points
                mask = self.cluster_labels == label
                plt.scatter(
                    self.pca_features[mask, 0],
                    self.pca_features[mask, 1],
                    c='gray',
                    label='Noise'
                )
            else:
                # Plot cluster points
                mask = self.cluster_labels == label
                plt.scatter(
                    self.pca_features[mask, 0],
                    self.pca_features[mask, 1],
                    label=f'Cluster {label}'
                )
        
        # Add labels
        for i, token in enumerate(self.tokens):
            if not token.log_returns.empty:
                plt.annotate(
                    token.symbol,
                    (self.pca_features[i, 0], self.pca_features[i, 1])
                )
        
        plt.title('DBSCAN Clustering of Tokens')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True)
        plt.show()

    def make_pairs(self) -> List[List[Token]]:
        """Create pairs from the clusters"""
        pairs = []
        
        for cluster in self.clusters:
            # Create all possible pairs within the cluster
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    pairs.append([cluster[i], cluster[j]])
        
        return pairs

    def get_cluster_info(self) -> dict:
        """Get information about the clustering results"""
        if self.cluster_labels is None:
            raise ValueError("Run clustering first")
        
        info = {
            'n_clusters': len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0),
            'n_noise': list(self.cluster_labels).count(-1),
            'cluster_sizes': {},
            'explained_variance': self.pca_model.explained_variance_ratio_ if self.pca_model else None
        }
        
        for label in set(self.cluster_labels):
            if label != -1:
                info['cluster_sizes'][f'cluster_{label}'] = list(self.cluster_labels).count(label)
        
        return info 