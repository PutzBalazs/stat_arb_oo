from stat_arb.strategy.strat_arb import StatArb
import logging

def main():
    # Initialize StatArb
    stat_arb = StatArb()
    
    # Load or fetch tokens
    tokens = stat_arb.load_or_fetch_tokens(chain_id=42161, max_tokens=10)
    
    # Run clustering
    clustering_results = stat_arb.run_clustering()
    
    # Find cointegrated pairs
    cointegrated_pairs = stat_arb.find_cointegrated_pairs()
    
    # Print summary of cointegrated pairs
    stat_arb.print_cointegrated_pairs_summary()
    
    # Run backtest on the first cointegrated pair if available
    if cointegrated_pairs:
        first_pair = cointegrated_pairs[0]
        stat_arb.run_backtest(
            pair=first_pair,
            entry_std=1.0,
            exit_std=0.0,
            initial_capital=1000.0
        )

if __name__ == "__main__":
    main()

