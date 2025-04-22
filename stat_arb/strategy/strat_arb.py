from ..core.token import Token
from ..core.pair import Pair
from ..core.coint_pair import CointPair
from ..core.backtester import Backtester
from ..clusterer import DBSCANClusterer, KMeansClusterer
from ..storage import JoblibDataStorage
from ..dex import OneInchDex
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Optional

class StatArb:
    def __init__(self, data_path: str = "data/"):
        """Initialize StatArb with data storage path"""
        self.data_storage = JoblibDataStorage(base_path=data_path)
        self.tokens: List[Token] = []
        self.coint_pairs: List[CointPair] = []
        self.clustering_results: Dict = {}
        self.backtest_results: Dict = {}

    def load_or_fetch_tokens(self, chain_id: int = 42161, max_tokens: int = 10) -> List[Token]:
        """Load existing tokens or fetch new ones from DEX"""
        try:
            print("Attempting to load existing tokens...")
            self.tokens = self.data_storage.read("dex", "oneinch_tokens")
            print("Successfully loaded tokens from storage")
        except Exception as e:
            print(f"Could not load tokens: {e}")
            print("Fetching new data...")
            
            # If loading fails, fetch new data
            dex = OneInchDex(chain_id=chain_id, max_tokens=max_tokens)
            self.tokens = dex.fetch_and_prepare_data()
            logging.info(f"Fetched {len(self.tokens)} tokens")
            
            # Save the new tokens
            self.data_storage.save(self.tokens, "dex", "oneinch_tokens")
            print("Saved new tokens to storage")
        
        return self.tokens

    def print_tokens_info(self):
        """Print information about all tokens"""
        print("\nToken Information:")
        print("-" * 50)
        for token in self.tokens:
            print(token)
            print("-" * 50)

    def test_token_functions(self, token: Token):
        """Test all functions and properties of a token"""
        print("\nTesting Token Functions and Properties:")
        print("=" * 50)
        
        # Basic Properties
        print("\nBasic Properties:")
        print(f"Address: {token.address}")
        print(f"Name: {token.name}")
        print(f"Symbol: {token.symbol}")
        print(f"Decimals: {token.decimals}")
        print(f"Logo: {token.logo}")
        
        # Data Properties
        print("\nData Properties:")
        print("OHLC Data Shape:", token.ohlc_data.shape if not token.ohlc_data.empty else "No data")
        print("Close Prices Head:", token.close_prices.head() if not token.close_prices.empty else "No data")
        print("Normalized Prices Head:", token.normalized_prices.head() if not token.normalized_prices.empty else "No data")
        print("Log Returns Head:", token.log_returns.head() if not token.log_returns.empty else "No data")
        
        # Calculated Properties
        print("\nCalculated Properties:")
        print("Log Returns (calc):", token.calc_log_returns()[:5] if len(token.calc_log_returns()) > 0 else "No data")
        print("Normalized Prices (calc):", token.normalize().head() if not token.normalize().empty else "No data")
        
        # Info Method
        print("\nInfo Method:")
        info = token.info()
        for key, value in info.items():
            print(f"{key}: {value}")
        
        # String Representation
        print("\nString Representation:")
        print(str(token))
        
        # Visualization
        print("\nVisualization Tests (plots will be shown):")
        token.visualize(plot_type='price')
        token.visualize(plot_type='normalized')
        token.visualize(plot_type='returns')

    def run_clustering(self):
        """Run both DBSCAN and KMeans clustering on tokens"""
        print("\nRunning Clustering Algorithms:")
        print("=" * 50)
        
        # Run DBSCAN Clustering
        print("\nRunning DBSCAN Clustering:")
        print("-" * 30)
        try:
            dbscan = DBSCANClusterer(
                tokens=self.tokens,
                n_components=2,
                eps=0.5,
                min_samples=2
            )
            
            dbscan_clusters = dbscan.run()
            dbscan_info = dbscan.get_cluster_info()
            dbscan_pairs = dbscan.make_pairs()
            
            print("\nDBSCAN Results:")
            print(f"Number of clusters: {dbscan_info['n_clusters']}")
            print(f"Number of noise points: {dbscan_info['n_noise']}")
            print("\nCluster Sizes:", dbscan_info['cluster_sizes'])
            print("\nExplained Variance:", dbscan_info['explained_variance'])
            
            # Visualize DBSCAN clusters
            dbscan.visualize()
            
        except Exception as e:
            print(f"Error in DBSCAN clustering: {str(e)}")
            dbscan_clusters = []
            dbscan_pairs = []
        
        # Run KMeans Clustering
        print("\nRunning KMeans Clustering:")
        print("-" * 30)
        try:
            kmeans = KMeansClusterer(
                tokens=self.tokens,
                n_components=2,
                n_clusters=3,
                random_state=42
            )
            
            kmeans_clusters = kmeans.run()
            kmeans_info = kmeans.get_cluster_info()
            kmeans_pairs = kmeans.make_pairs()
            
            print("\nKMeans Results:")
            print(f"Number of clusters: {kmeans_info['n_clusters']}")
            print("\nCluster Sizes:", kmeans_info['cluster_sizes'])
            print("\nExplained Variance:", kmeans_info['explained_variance'])
            
            # Visualize KMeans clusters
            kmeans.visualize()
            
        except Exception as e:
            print(f"Error in KMeans clustering: {str(e)}")
            kmeans_clusters = []
            kmeans_pairs = []
        
        # Save clustering results
        self.clustering_results = {
            'dbscan_clusters': dbscan_clusters,
            'dbscan_pairs': dbscan_pairs,
            'kmeans_clusters': kmeans_clusters,
            'kmeans_pairs': kmeans_pairs
        }
        
        self.data_storage.save(self.clustering_results, "clustering", "clustering_results")
        print("\nSaved clustering results to storage")
        
        return self.clustering_results

    def process_cluster_pairs(self, pairs: List[Pair], z_score_window: int = 20) -> List[CointPair]:
        """Process pairs from a cluster to find cointegrated pairs"""
        cointegrated_pairs = []
        
        print(f"\nProcessing {len(pairs)} pairs...")
        for pair in pairs:
            coint_pair = CointPair(pair.token1, pair.token2, z_score_window)
            
            if coint_pair.is_cointegrated:
                print(f"\nFound cointegrated pair: {coint_pair}")
                print("Cointegration stats:", coint_pair.cointegration_stats)
                print("Current z-score:", coint_pair.zscore.iloc[-1] if coint_pair.zscore is not None else None)
                print("Trade signal:", coint_pair.get_trade_signal())
                
                cointegrated_pairs.append(coint_pair)
        
        return cointegrated_pairs

    def find_cointegrated_pairs(self):
        """Find cointegrated pairs from clustering results"""
        if not self.clustering_results:
            print("No clustering results available. Run clustering first.")
            return []
        
        all_cointegrated_pairs = []
        
        # Process KMeans pairs
        print("\nProcessing KMeans pairs...")
        kmeans_coint_pairs = self.process_cluster_pairs(self.clustering_results['kmeans_pairs'])
        all_cointegrated_pairs.extend(kmeans_coint_pairs)
        
        # Process DBSCAN pairs
        print("\nProcessing DBSCAN pairs...")
        dbscan_coint_pairs = self.process_cluster_pairs(self.clustering_results['dbscan_pairs'])
        all_cointegrated_pairs.extend(dbscan_coint_pairs)
        
        print(f"\nFound total of {len(all_cointegrated_pairs)} cointegrated pairs")
        
        # Save cointegrated pairs
        self.coint_pairs = all_cointegrated_pairs
        self.data_storage.save(self.coint_pairs, "pairs", "cointegrated_pairs")
        print("Saved cointegrated pairs to storage")
        
        return self.coint_pairs

    def print_cointegrated_pairs_summary(self):
        """Print summary of cointegrated pairs"""
        print("\nCointegrated Pairs Summary:")
        for pair in self.coint_pairs:
            print(f"\n{pair}")
            print("Stats:", pair.cointegration_stats)
            print("Current signal:", pair.get_trade_signal())

    def run_backtest(self, pair: CointPair, entry_std: float = 1.0, exit_std: float = 0.0, initial_capital: float = 1000.0):
        """Run backtest on a cointegrated pair"""
        print(f"\nRunning backtest on pair: {pair}")
        
        backtester = Backtester(pair)
        results = backtester.run(entry_std=entry_std, exit_std=exit_std, initial_capital=initial_capital)
        
        if results.success:
            print("\nBacktest Results:")
            print(f"Initial Capital: ${results.initial_capital:.2f}")
            print(f"Final Capital: ${results.final_capital:.2f}")
            print(f"Total Return: {((results.final_capital/results.initial_capital)-1):.2%}")
            print(f"Total Trades: {results.total_trades}")
            print(f"Win Rate: {results.win_rate:.2%}")
            print(f"Total PnL: ${results.total_pnl:.2f}")
            print(f"Avg PnL: ${results.avg_pnl:.2f}")
            print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
            print(f"Max Drawdown: {results.max_drawdown:.2%}")
            
            # Visualize the backtest results
            backtester.visualize(entry_std=entry_std, exit_std=exit_std, initial_capital=initial_capital)
        else:
            print(f"Backtest failed: {results.error}")
        
        return results
