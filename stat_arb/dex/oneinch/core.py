from . import fetch_utils
from stat_arb.core.token import Token
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

class OneInchDex:
    def __init__(self, chain_id=1, granularity="day", limit=30, api_wait=1, max_tokens=100):
        self.chain_id = chain_id
        self.granularity = granularity
        self.limit = limit
        self.api_wait = api_wait
        self.max_tokens = max_tokens
        self._tokens = None
        
    def fetch_and_prepare_data(self):
        print("Fetching tokens...")
        all_tokens = fetch_utils.get_all_tokens(self.chain_id)
        print(f"Found {len(all_tokens)} tokens")

        print("Extracting basic token info...")
        token_data = fetch_utils.extract_basic_token_info(all_tokens, self.max_tokens)
        print(f"Selected {len(token_data)} tokens for analysis")

        print("Fetching OHLC data...")
        token_data = fetch_utils.fetch_ohlc_for_tokens(
            token_data, self.chain_id, self.granularity, self.limit, self.api_wait
        )
        print(f"Got price data for {len(token_data)} tokens")

        # Create Token objects
        print("Creating Token objects...")
        self._tokens = []
        for address, data in token_data.items():
            token = Token(address, data)
            self._tokens.append(token)
        
        return self._tokens

    def get_log_returns_matrix(self) -> pd.DataFrame:
        """Get a matrix of log returns for all tokens"""
        if not self._tokens:
            raise ValueError("No tokens available. Call fetch_and_prepare_data() first.")
        
        # Create a DataFrame with timestamps as index
        returns_df = pd.DataFrame()
        
        for token in self._tokens:
            if not token.log_returns.empty:
                returns_df[token.symbol] = token.log_returns
        
        return returns_df

    def perform_pca(self, n_components: int = 2) -> dict:
        """Perform PCA on the log returns of all tokens"""
        if not self._tokens:
            raise ValueError("No tokens available. Call fetch_and_prepare_data() first.")
        
        # Get log returns matrix
        returns_df = self.get_log_returns_matrix()
        
        if returns_df.empty:
            raise ValueError("No valid returns data available")
        
        # Drop any rows with NaN values
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 2:
            raise ValueError("Insufficient data points for PCA")
        
        # Perform PCA
        pca = PCA(n_components=min(n_components, len(returns_df.columns)))
        pca_result = pca.fit_transform(returns_df)
        
        # Create results dictionary
        results = {
            'components': pca.components_,
            'explained_variance': pca.explained_variance_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'transformed_data': pca_result,
            'feature_names': returns_df.columns.tolist()
        }
        
        return results

    def get_pca_analysis(self, n_components: int = 2) -> dict:
        """Get detailed PCA analysis including component loadings"""
        pca_results = self.perform_pca(n_components)
        
        # Create component loadings DataFrame
        loadings = pd.DataFrame(
            pca_results['components'],
            columns=pca_results['feature_names'],
            index=[f'PC{i+1}' for i in range(len(pca_results['components']))]
        )
        
        # Create variance explained DataFrame
        variance = pd.DataFrame({
            'Explained Variance': pca_results['explained_variance'],
            'Explained Variance Ratio': pca_results['explained_variance_ratio'],
            'Cumulative Ratio': np.cumsum(pca_results['explained_variance_ratio'])
        }, index=[f'PC{i+1}' for i in range(len(pca_results['components']))])
        
        return {
            'loadings': loadings,
            'variance': variance,
            'transformed_data': pca_results['transformed_data']
        }

    