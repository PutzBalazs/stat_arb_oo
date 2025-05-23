from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from stat_arb.core import Token, Pair
from stat_arb.clusterer.clusterer import Clusterer

class DBSCANClusterer(Clusterer):
    """DBSCAN-based token clustering implementation"""
    
    def __init__(
        self,
        tokens: List[Token],
        n_components: int = 2,
        eps: float = 0.5,
        min_samples: int = 2
    ):
        super().__init__(tokens, n_components)
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    
    def run(self) -> List[List[Token]]:
        """Run DBSCAN clustering on token data"""
        # Prepare features using parent class method
        features = self._prepare_features()
        
        # Perform PCA
        self.pca = PCA(n_components=self.n_components)
        pca_features = self.pca.fit_transform(features.T)  # Transpose to get tokens as rows
        
        # Run DBSCAN clustering
        self.labels = self.dbscan.fit_predict(pca_features)
        
        # Group tokens by cluster
        self.clusters = []
        unique_labels = np.unique(self.labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
            cluster_tokens = [self.tokens[i] for i in range(len(self.tokens)) if self.labels[i] == label]
            self.clusters.append(cluster_tokens)
        
        return self.clusters
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about the DBSCAN clustering results"""
        if self.labels is None:
            raise ValueError("Clustering must be run before getting cluster info")
        
        unique_labels = np.unique(self.labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Exclude noise points
        n_noise = np.sum(self.labels == -1)
        
        cluster_sizes = {}
        for label in unique_labels:
            if label == -1:
                cluster_sizes['noise'] = np.sum(self.labels == -1)
            else:
                cluster_sizes[f'cluster_{label}'] = np.sum(self.labels == label)
        
        return {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'cluster_sizes': cluster_sizes,
            'explained_variance': self.pca.explained_variance_ratio_.tolist() if self.pca else None
        }
    
    def make_pairs(self) -> List[Pair]:
        """Create trading pairs from DBSCAN clusters"""
        if self.clusters is None:
            raise ValueError("Clustering must be run before creating pairs")
        
        pairs = []
        for cluster in self.clusters:
            # Create pairs within each cluster
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    pairs.append(Pair(cluster[i], cluster[j]))
        
        return pairs 