from ..core.token import Token
from ..core.coint_pair import CointPair
from ..cluster.clusterer import Clusterer

class StatArb:
    def __init__(self):
        self.tokens = []
        self.coint_pairs = []
        self.trades = []

    def make_tokens(self):
        pass

    def cluster(self):
        clusterer = Clusterer(self.tokens)
        return clusterer.run()

    def check_coint(self):
        pass

    def store_coint_pairs(self):
        pass

    def check_pairs_for_trade(self):
        pass

    def store_trades(self):
        pass

    def execute_trades(self):
        pass

    def update(self):
        pass
