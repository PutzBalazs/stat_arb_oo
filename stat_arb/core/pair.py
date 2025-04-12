import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from .token import Token

class Pair:
    def __init__(self, token1: Token, token2: Token):
        self.token1 = token1
        self.token2 = token2
        self._spread = None
        self._hedge_ratio = None
        self._is_cointegrated = None

    @property
    def symbols(self) -> Tuple[str, str]:
        """Get the symbols of both tokens"""
        return (self.token1.symbol, self.token2.symbol)

    @property
    def spread(self) -> Optional[pd.Series]:
        """Calculate and return the price spread"""
        if self._spread is None and not self.token1.close_prices.empty and not self.token2.close_prices.empty:
            # Align the price series
            aligned_prices = pd.concat([self.token1.close_prices, self.token2.close_prices], axis=1).dropna()
            if not aligned_prices.empty:
                self._spread = aligned_prices.iloc[:, 0] - aligned_prices.iloc[:, 1]
        return self._spread

    @property
    def hedge_ratio(self) -> Optional[float]:
        """Calculate and return the hedge ratio"""
        if self._hedge_ratio is None and self.spread is not None:
            # Simple hedge ratio calculation (can be enhanced with OLS)
            self._hedge_ratio = 1.0  # Placeholder for now
        return self._hedge_ratio

    def check_cointegration(self) -> bool:
        """Check if the pair is cointegrated"""
        if self._is_cointegrated is None and self.spread is not None:
            # Placeholder for cointegration test
            # TODO: Implement proper cointegration test (e.g., ADF test)
            self._is_cointegrated = True
        return self._is_cointegrated

    def visualize(self, plot_type: str = 'spread'):
        """Visualize the pair relationship"""
        if plot_type == 'spread' and self.spread is not None:
            # Standardize the spread
            standardized_spread = (self.spread - self.spread.mean()) / self.spread.std()
            
            plt.figure(figsize=(12, 6))
            standardized_spread.plot(title=f'Standardized Spread: {self.token1.symbol} - {self.token2.symbol}')
            plt.axhline(y=0, color='r', linestyle='--', label='Mean')
            plt.axhline(y=2, color='g', linestyle='--', alpha=0.5, label='+2σ')
            plt.axhline(y=-2, color='g', linestyle='--', alpha=0.5, label='-2σ')
            plt.legend()
            plt.grid(True)
            plt.show()
        elif plot_type == 'prices':
            # Normalize prices for better comparison
            norm_token1 = self.token1.close_prices / self.token1.close_prices.iloc[0]
            norm_token2 = self.token2.close_prices / self.token2.close_prices.iloc[0]
            
            plt.figure(figsize=(12, 6))
            norm_token1.plot(label=f'{self.token1.symbol} (normalized)')
            norm_token2.plot(label=f'{self.token2.symbol} (normalized)')
            plt.title(f'Normalized Price Comparison: {self.token1.symbol} vs {self.token2.symbol}')
            plt.legend()
            plt.grid(True)
            plt.show()

    def __str__(self) -> str:
        return f"Pair({self.token1.symbol}-{self.token2.symbol})"

    def info(self) -> dict:
        """Get information about the pair"""
        return {
            'symbols': self.symbols,
            'spread_available': self.spread is not None,
            'hedge_ratio': self.hedge_ratio,
            'is_cointegrated': self.check_cointegration()
        }
