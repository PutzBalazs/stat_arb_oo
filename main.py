from stat_arb.strategy.strat_arb import StatArb
import logging
from stat_arb.dex.oneinch.core import OneInchDex

def main():
    stat_arb = StatArb()
    tokens = stat_arb.load_or_fetch_tokens(chain_id=42161, max_tokens=100)
    clustering_results = stat_arb.run_clustering()    
    cointegrated_pairs = stat_arb.find_cointegrated_pairs(load_from_storage=False)
    
    #stat_arb.print_cointegrated_pairs_summary()
    
    # order based on zero crossings
    cointegrated_pairs.sort(key=lambda x: x.cointegration_stats['zero_crossings'], reverse=True)
    
    # visulize_speread for top 5 most zero crossings
    for pair in cointegrated_pairs[:20]:
        pair.visualize_spread()
        

def main_1():
    """Example of executing a WETH to DAI swap"""
    # Initialize OneInchDex
    dex = OneInchDex()
    
    # Execute swap
    src_token = "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1"  # WETH
    dst_token = "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1"  # DAI
    amount = 0.0005  # 0.1 WETH
    
    try:
        tx_hash = dex.execute_swap(src_token, dst_token, amount)
        print(f"Swap executed successfully! Transaction hash: {tx_hash}")
    except Exception as e:
        print(f"Failed to execute swap: {str(e)}")

if __name__ == "__main__":
    main()  # Run the trading example
