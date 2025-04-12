from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from typing import List, Dict, Any
from ..core import Token, Pair

class Clusterer(ABC):
    """Abstract base class for clustering algorithms"""
    
    def __init__(self, tokens: List[Token], n_components: int = 2):
        self.tokens = tokens
        self.n_components = n_components
        self.pca = None
        self.features = None
        self.clusters = None
        self.labels = None
    
    def _prepare_features(self) -> pd.DataFrame:
        """Prepare features for clustering by combining token data"""
        if not self.tokens:
            raise ValueError("No tokens provided for clustering")
        
        # Get normalized prices for all tokens
        normalized_prices = []
        for token in self.tokens:
            if token.normalized_prices is not None:
                normalized_prices.append(token.normalized_prices)
        
        if not normalized_prices:
            raise ValueError("No normalized price data available for clustering")
        
        # Combine all normalized prices into a single DataFrame
        features = pd.concat(normalized_prices, axis=1)
        features.columns = [token.symbol for token in self.tokens]
        
        # Fill any missing values
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        # Store the features
        self.features = features
        return features
    
    @abstractmethod
    def run(self) -> List[List[Token]]:
        """Run the clustering algorithm and return list of clusters"""
        pass
    
    @abstractmethod
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about the clustering results"""
        pass
    
    @abstractmethod
    def make_pairs(self) -> List[Pair]:
        """Create trading pairs from clusters"""
        pass
    
    def visualize(self, title: str = None) -> None:
        """Visualize the clustering results using PCA components"""
        if self.pca is None or self.labels is None:
            raise ValueError("Clustering must be run before visualization")
        
        import matplotlib.pyplot as plt
        
        # Get PCA components
        pca_components = self.pca.transform(self.features.T)  # Transpose to get tokens as rows
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        
        # Get unique labels and create a color map
        unique_labels = np.unique(self.labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        
        # Plot each cluster
        for i, label in enumerate(unique_labels):
            mask = self.labels == label
            color = colors[i]
            if label == -1:  # Special case for noise points in DBSCAN
                plt.scatter(
                    pca_components[mask, 0],
                    pca_components[mask, 1],
                    c='gray',
                    label='Noise' if label == -1 else f'Cluster {label}',
                    s=100
                )
            else:
                plt.scatter(
                    pca_components[mask, 0],
                    pca_components[mask, 1],
                    c=[color],
                    label=f'Cluster {label}',
                    s=100
                )
        
        # Add token symbols as labels
        for i, token in enumerate(self.tokens):
            plt.annotate(
                token.symbol,
                (pca_components[i, 0], pca_components[i, 1]),
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        # Add explained variance to axis labels
        explained_variance = self.pca.explained_variance_ratio_
        plt.xlabel(f'PC1 ({explained_variance[0]:.2%})')
        plt.ylabel(f'PC2 ({explained_variance[1]:.2%})')
        
        # Add title
        if title:
            plt.title(title)
        else:
            plt.title(f'Token Clustering Results ({self.__class__.__name__})')
        
        plt.legend(title='Cluster')
        plt.grid(True)
        plt.show()
