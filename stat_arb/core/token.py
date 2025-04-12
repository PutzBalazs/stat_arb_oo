import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from lightweight_charts import Chart

class Token:
    def __init__(self, address: str, data: Dict):
        self._address = address
        self._basic_info = data.get('basic_info', {})
        self._ohlc_data = data.get('ohlc_data', pd.DataFrame())
        
        # Initialize properties
        self._normalized_prices = None
        self._log_returns = None

    @property
    def address(self) -> str:
        return self._address

    @property
    def name(self) -> str:
        return self._basic_info.get('name', 'Unknown')

    @property
    def symbol(self) -> str:
        return self._basic_info.get('symbol', 'UNK')

    @property
    def decimals(self) -> int:
        return self._basic_info.get('decimals', 18)

    @property
    def logo(self) -> str:
        return self._basic_info.get('logo', '')

    @property
    def ohlc_data(self) -> pd.DataFrame:
        return self._ohlc_data

    @property
    def close_prices(self) -> pd.Series:
        return self._ohlc_data['close'] if not self._ohlc_data.empty else pd.Series()

    @property
    def normalized_prices(self) -> pd.Series:
        if self._normalized_prices is None and not self.close_prices.empty:
            self._normalized_prices = self.close_prices / self.close_prices.iloc[0]
        return self._normalized_prices

    @property
    def log_returns(self) -> pd.Series:
        if self._log_returns is None and not self.close_prices.empty:
            self._log_returns = np.log(self.close_prices / self.close_prices.shift(1))
        return self._log_returns

    def calc_log_returns(self) -> np.ndarray:
        """Calculate log returns from the data"""
        if not self.close_prices.empty:
            return np.log(self.close_prices / self.close_prices.shift(1)).values
        return np.array([])

    def normalize(self) -> pd.Series:
        """Normalize the price data"""
        if not self.close_prices.empty:
            return self.close_prices / self.close_prices.iloc[0]
        return pd.Series()


    def visualize(self, plot_type: str = 'price'):
        """Visualize the token data"""
        if plot_type == 'kline':
            self.visualize_kline()
            return
            
        import matplotlib.pyplot as plt
        
        if self.close_prices.empty:
            print("No data available for visualization")
            return

        plt.figure(figsize=(10, 6))
        if plot_type == 'price':
            plt.plot(self.close_prices.index, self.close_prices.values, label='Price')
        elif plot_type == 'normalized':
            plt.plot(self.normalized_prices.index, self.normalized_prices.values, label='Normalized Price')
        elif plot_type == 'returns':
            plt.plot(self.log_returns.index, self.log_returns.values, label='Log Returns')
        
        plt.title(f"{self.symbol} ({self.name})")
        plt.xlabel('Date')
        plt.ylabel(plot_type.capitalize())
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_kline(self):
        """Visualize the token data using candlestick chart"""
        if self.ohlc_data.empty:
            print("No OHLC data available for visualization")
            return
        data = self.ohlc_data.copy().sort_values('timestamp', ascending=True).reset_index(drop=True)
        data['time'] = pd.to_datetime(data['timestamp']).dt.strftime('%Y-%m-%d')
        data = data[['time', 'open', 'high', 'low', 'close']]
        data['volume'] = 0
        chart = Chart(toolbox=True)
        chart.set(data)
        chart.layout(background_color='#FFFACD', text_color='#000000')
        chart.grid(vert_enabled=False, horz_enabled=False)
        chart.show(block=True)

    def info(self) -> Dict:
        """Return basic information about the token"""
        return {
            'address': self.address,
            'name': self.name,
            'symbol': self.symbol,
            'decimals': self.decimals,
            'data_points': len(self.close_prices),
            'first_price': self.close_prices.iloc[0] if not self.close_prices.empty else None,
            'last_price': self.close_prices.iloc[-1] if not self.close_prices.empty else None,
            'returns_mean': self.log_returns.mean() if not self.log_returns.empty else None,
            'returns_std': self.log_returns.std() if not self.log_returns.empty else None
        }

    def __str__(self) -> str:
        """String representation of the token"""
        info = self.info()
        return (f"Token({self.symbol}):\n"
                f"  Name: {self.name}\n"
                f"  Address: {self.address}\n"
                f"  Data Points: {info['data_points']}\n"
                f"  Price Range: {info['first_price']:.4f} -> {info['last_price']:.4f}\n"
                f"  Returns (mean, std): ({info['returns_mean']:.4f}, {info['returns_std']:.4f})")

    def __repr__(self) -> str:
        return f"Token(address='{self.address}', symbol='{self.symbol}')"
