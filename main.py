from stat_arb.storage import JoblibDataStorage
from stat_arb.dex import OneInchDex
from stat_arb.core import Token
from stat_arb.clusterer import DBSCANClusterer

def print_tokens_info(tokens):
    """Print information about all tokens"""
    print("\nToken Information:")
    print("-" * 50)
    for token in tokens:
        print(token)
        print("-" * 50)

def test_token_functions(token):
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

def test_clustering(tokens):
    """Test DBSCAN clustering on tokens"""
    print("\nTesting DBSCAN Clustering:")
    print("=" * 50)
    
    try:
        # Initialize clusterer with different parameters
        clusterer = DBSCANClusterer(
            tokens=tokens,
            n_components=2,
            eps=0.5,
            min_samples=2
        )
        
        # Run clustering
        clusters = clusterer.run()
        
        # Get cluster info
        info = clusterer.get_cluster_info()
        
        # Print clustering results
        print("\nClustering Results:")
        print(f"Number of clusters: {info['n_clusters']}")
        print(f"Number of noise points: {info['n_noise']}")
        print("\nCluster Sizes:")
        for cluster, size in info['cluster_sizes'].items():
            print(f"{cluster}: {size} tokens")
        
        print("\nExplained Variance:")
        print(info['explained_variance'])
        
        # Print tokens in each cluster
        print("\nTokens in Clusters:")
        for i, cluster in enumerate(clusters):
            print(f"\nCluster {i}:")
            for token in cluster:
                print(f"  - {token.symbol}")
        
        # Create and print pairs
        pairs = clusterer.make_pairs()
        print(f"\nNumber of pairs created: {len(pairs)}")
        print("\nFirst 5 pairs:")
        for i, pair in enumerate(pairs[:5]):
            print(f"Pair {i+1}: {pair[0].symbol} - {pair[1].symbol}")
        
        # Visualize clusters
        print("\nVisualizing clusters...")
        clusterer.visualize()
        
    except Exception as e:
        print(f"Error in clustering: {str(e)}")

def main():
    # Initialize JolibDataStorage
    data_storage = JoblibDataStorage(base_path="data/")
    
    # Try to load existing tokens first
    try:
        print("Attempting to load existing tokens...")
        tokens = data_storage.read("dex", "oneinch_tokens")
        print("Successfully loaded tokens from storage")
    except Exception as e:
        print(f"Could not load tokens: {e}")
        print("Fetching new data...")
        
        # If loading fails, fetch new data
        dex = OneInchDex(chain_id=42161, max_tokens=10)
        tokens = dex.fetch_and_prepare_data()
        
        # Save the new tokens
        data_storage.save(tokens, "dex", "oneinch_tokens")
        print("Saved new tokens to storage")
    
    # Print all tokens info
    print_tokens_info(tokens)
    
    # Test functions on first token if available
    if tokens:
        print("\nTesting first token:")
        test_token_functions(tokens[0])        
        # Test clustering
        test_clustering(tokens)
    else:
        print("No tokens available for testing")

if __name__ == "__main__":
    main()
