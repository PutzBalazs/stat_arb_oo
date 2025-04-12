from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from stat_arb.core import Token, Pair
from stat_arb.clusterer.clusterer import Clusterer

class KMeansClusterer(Clusterer):
    """KMeans-based token clustering implementation"""
    
    def __init__(
        self,
        tokens: List[Token],
        n_components: int = 2,
        n_clusters: int = 3,
        random_state: int = 42
    ):
        super().__init__(tokens, n_components)
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    
    def run(self) -> List[List[Token]]:
        """Run KMeans clustering on token data"""
        # Prepare features using parent class method
        features = self._prepare_features()
        
        # Perform PCA
        self.pca = PCA(n_components=self.n_components)
        pca_features = self.pca.fit_transform(features.T)  # Transpose to get tokens as rows
        
        # Run KMeans clustering
        self.labels = self.kmeans.fit_predict(pca_features)
        
        # Group tokens by cluster
        self.clusters = []
        for i in range(self.n_clusters):
            cluster_tokens = [self.tokens[j] for j in range(len(self.tokens)) if self.labels[j] == i]
            self.clusters.append(cluster_tokens)
        
        return self.clusters
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about the KMeans clustering results"""
        if self.labels is None:
            raise ValueError("Clustering must be run before getting cluster info")
        
        cluster_sizes = {}
        for i in range(self.n_clusters):
            cluster_sizes[f'cluster_{i}'] = np.sum(self.labels == i)
        
        return {
            'n_clusters': self.n_clusters,
            'cluster_sizes': cluster_sizes,
            'explained_variance': self.pca.explained_variance_ratio_.tolist() if self.pca else None
        }
    
    def make_pairs(self) -> List[Pair]:
        """Create trading pairs from KMeans clusters"""
        if self.clusters is None:
            raise ValueError("Clustering must be run before creating pairs")
        
        pairs = []
        for cluster in self.clusters:
            # Create pairs within each cluster
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    pairs.append(Pair(cluster[i], cluster[j]))
        
        return pairs 