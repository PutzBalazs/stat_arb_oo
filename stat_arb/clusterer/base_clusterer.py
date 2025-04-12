from abc import ABC, abstractmethod
from typing import List
from ..core import Token

class Clusterer(ABC):
    @abstractmethod
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.clusters = []

    @abstractmethod
    def run(self) -> List[List[Token]]:
        """Run the clustering algorithm and return list of token clusters"""
        pass

    @abstractmethod
    def visualize(self):
        """Visualize the clustering results"""
        pass

    @abstractmethod
    def make_pairs(self) -> List[List[Token]]:
        """Create pairs from the clusters"""
        pass 