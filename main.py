from stat_arb.storage import JoblibDataStorage
from stat_arb.dex import OneInchDex
from stat_arb.core import Token, Pair
from stat_arb.clusterer import DBSCANClusterer, KMeansClusterer

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
    """Test both DBSCAN and KMeans clustering on tokens"""
    print("\nTesting Clustering Algorithms:")
    print("=" * 50)
    
    # Test DBSCAN Clustering
    print("\nTesting DBSCAN Clustering:")
    print("-" * 30)
    try:
        # Initialize DBSCAN clusterer
        dbscan = DBSCANClusterer(
            tokens=tokens,
            n_components=2,
            eps=0.5,
            min_samples=2
        )
        
        # Run clustering
        dbscan_clusters = dbscan.run()
        
        # Get cluster info
        dbscan_info = dbscan.get_cluster_info()
        
        # Print clustering results
        print("\nDBSCAN Results:")
        print(f"Number of clusters: {dbscan_info['n_clusters']}")
        print(f"Number of noise points: {dbscan_info['n_noise']}")
        print("\nCluster Sizes:")
        for cluster, size in dbscan_info['cluster_sizes'].items():
            print(f"{cluster}: {size} tokens")
        
        print("\nExplained Variance:")
        print(dbscan_info['explained_variance'])
        
        # Print tokens in each cluster
        print("\nTokens in DBSCAN Clusters:")
        for i, cluster in enumerate(dbscan_clusters):
            print(f"\nCluster {i}:")
            for token in cluster:
                print(f"  - {token.symbol}")
        
        # Visualize DBSCAN clusters
        print("\nVisualizing DBSCAN clusters...")
        dbscan.visualize()
        # Create and test pairs
        dbscan_pairs = dbscan.make_pairs()
        print(f"\nNumber of DBSCAN pairs created: {len(dbscan_pairs)}")
        print("\nTesting first 5 DBSCAN pairs:")
        for i, pair in enumerate(dbscan_pairs[:5]):
            print(f"\nPair {i+1}: {pair}")
            print("Pair Info:")
            pair_info = pair.info()
            for key, value in pair_info.items():
                print(f"  {key}: {value}")
            pair.visualize(plot_type='prices')
            pair.visualize(plot_type='spread')
      
        
    except Exception as e:
        print(f"Error in DBSCAN clustering: {str(e)}")
    
    # Test KMeans Clustering
    print("\nTesting KMeans Clustering:")
    print("-" * 30)
    try:
        # Initialize KMeans clusterer
        kmeans = KMeansClusterer(
            tokens=tokens,
            n_components=2,
            n_clusters=3,
            random_state=42
        )
        
        # Run clustering
        kmeans_clusters = kmeans.run()
        
        # Get cluster info
        kmeans_info = kmeans.get_cluster_info()
        
        # Print clustering results
        print("\nKMeans Results:")
        print(f"Number of clusters: {kmeans_info['n_clusters']}")
        print("\nCluster Sizes:")
        for cluster, size in kmeans_info['cluster_sizes'].items():
            print(f"{cluster}: {size} tokens")
        
        print("\nExplained Variance:")
        print(kmeans_info['explained_variance'])
        
        # Print tokens in each cluster
        print("\nTokens in KMeans Clusters:")
        for i, cluster in enumerate(kmeans_clusters):
            print(f"\nCluster {i}:")
            for token in cluster:
                print(f"  - {token.symbol}")
        
        # Visualize KMeans clusters
        print("\nVisualizing KMeans clusters...")
        kmeans.visualize()
        # Create and test pairs
        kmeans_pairs = kmeans.make_pairs()
        print(f"\nNumber of KMeans pairs created: {len(kmeans_pairs)}")
        print("\nTesting first 5 KMeans pairs:")
        for i, pair in enumerate(kmeans_pairs[:5]):
            print(f"\nPair {i+1}: {pair}")
            print("Pair Info:")
            pair_info = pair.info()
            for key, value in pair_info.items():
                print(f"  {key}: {value}")
            pair.visualize(plot_type='prices')
            pair.visualize(plot_type='spread')
            
        
        
        # Save clusters and pairs
        data_storage = JoblibDataStorage(base_path="data/")
        data_storage.save({
            'dbscan_clusters': dbscan_clusters,
            'dbscan_pairs': dbscan_pairs,
            'kmeans_clusters': kmeans_clusters,
            'kmeans_pairs': kmeans_pairs
        }, "clustering", "clustering_results")
        print("\nSaved clustering results to storage")
        
    except Exception as e:
        print(f"Error in KMeans clustering: {str(e)}")

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
    """print_tokens_info(tokens)"""
    
    # Test functions on first token if available
    if tokens:
        #print("\nTesting first token:")
        #test_token_functions(tokens[0]) 
        first_token = tokens[0]
        print("starting kline")
        first_token.visualize(plot_type='kline')
        # Test clustering
        #test_clustering(tokens)
    else:
        print("No tokens available for testing")

if __name__ == "__main__":
    main()
