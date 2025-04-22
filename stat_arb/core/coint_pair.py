from .pair import Pair
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    position: str  # 'long' or 'short'
    pnl: Optional[float]
    entry_zscore: float
    exit_zscore: Optional[float]

class CointPair(Pair):
    def __init__(self, token1, token2, z_score_window: int = 20):
        super().__init__(token1, token2)
        self.z_score_window = z_score_window
        self._cointegration_results = None
        self._hedge_ratio = None
        self._spread = None
        self._zscore = None
        self._zero_crossings = None
        
        # Calculate all metrics on initialization
        self._calculate_all_metrics()
    
    def _calculate_all_metrics(self):
        """Calculate all statistical arbitrage metrics"""
        if self.token1.close_prices.empty or self.token2.close_prices.empty:
            return
            
        # Align price series
        aligned_prices = pd.concat([self.token1.close_prices, self.token2.close_prices], axis=1).dropna()
        series_1 = aligned_prices.iloc[:, 0]
        series_2 = aligned_prices.iloc[:, 1]
        
        # Calculate cointegration
        self._cointegration_results = self._calculate_cointegration(series_1, series_2)
        
        # Calculate spread and z-score if cointegrated
        if self.is_cointegrated:
            self._hedge_ratio = self._cointegration_results[4]  # hedge ratio from cointegration
            self._spread = self._calculate_spread(series_1, series_2, self._hedge_ratio)
            self._zscore = self._calculate_zscore(self._spread)
            self._zero_crossings = len(np.where(np.diff(np.sign(self._spread)))[0])
    
    def _calculate_cointegration(self, series_1: pd.Series, series_2: pd.Series) -> Tuple:
        """Calculate cointegration between two price series"""
        coint_flag = 0
        coint_res = coint(series_1, series_2)
        coint_t = coint_res[0]
        p_value = coint_res[1]
        critical_value = coint_res[2][1]
        
        # Calculate hedge ratio using OLS
        model = sm.OLS(series_1, series_2).fit()
        hedge_ratio = model.params.iloc[0]
        
        # Calculate spread
        spread = self._calculate_spread(series_1, series_2, hedge_ratio)
        zero_crossings = len(np.where(np.diff(np.sign(spread)))[0])
        
        # Determine cointegration
        if p_value < 0.5 and coint_t < critical_value:
            coint_flag = 1
            
        return (coint_flag, round(p_value, 2), round(coint_t, 2), 
                round(critical_value, 2), round(hedge_ratio, 2), zero_crossings)
    
    def _calculate_spread(self, series_1: pd.Series, series_2: pd.Series, hedge_ratio: float) -> pd.Series:
        """Calculate the spread between two price series"""
        return series_1 - (series_2 * hedge_ratio)
    
    def _calculate_zscore(self, spread: pd.Series) -> pd.Series:
        """Calculate the z-score of the spread"""
        # Fix: Ensure we're working with a 1D Series
        spread_series = pd.Series(spread)
        
        # Calculate rolling statistics
        mean = spread_series.rolling(window=self.z_score_window).mean()
        std = spread_series.rolling(window=self.z_score_window).std()
        
        # Calculate z-score
        zscore = (spread_series - mean) / std
        
        return zscore
    
    @property
    def is_cointegrated(self) -> bool:
        """Check if the pair is cointegrated"""
        if self._cointegration_results is None:
            return False
        return bool(self._cointegration_results[0])
    
    @property
    def cointegration_stats(self) -> dict:
        """Get cointegration statistics"""
        if self._cointegration_results is None:
            return {}
        return {
            'p_value': self._cointegration_results[1],
            'coint_t': self._cointegration_results[2],
            'critical_value': self._cointegration_results[3],
            'hedge_ratio': self._cointegration_results[4],
            'zero_crossings': self._cointegration_results[5]
        }
    
    @property
    def spread(self) -> Optional[pd.Series]:
        """Get the spread series"""
        return self._spread
    
    @property
    def zscore(self) -> Optional[pd.Series]:
        """Get the z-score series"""
        return self._zscore
    
    def get_trade_signal(self) -> Optional[str]:
        """Get current trading signal based on z-score"""
        if self._zscore is None or self._zscore.empty:
            return None
        
        # Fix: Properly extract the last value
        current_zscore = self._zscore.iloc[-1]
        if pd.isna(current_zscore):
            return None
        
        # Check for zero crossing (exit signal)
        if len(self._zscore) > 1:
            prev_zscore = self._zscore.iloc[-2]
            if (prev_zscore * current_zscore) < 0:  # Zero crossing
                return 'exit'
        
        # Entry signals
        if current_zscore <= -1.0:
            return 'long'
        elif current_zscore >= 1.0:
            return 'short'
        return None
    
    def info(self) -> dict:
        """Get comprehensive information about the pair"""
        base_info = super().info()
        current_zscore = None
        if self._zscore is not None and not self._zscore.empty:
            current_zscore = self._zscore.iloc[-1]
            if pd.isna(current_zscore):
                current_zscore = None
            
        coint_info = {
            'is_cointegrated': self.is_cointegrated,
            'cointegration_stats': self.cointegration_stats,
            'current_zscore': current_zscore,
            'trade_signal': self.get_trade_signal()
        }
        return {**base_info, **coint_info}

    def visualize_spread(self):
        """Visualize the price trends, spread, and z-score"""
        if self._spread is None or self._zscore is None:
            print("No spread or z-score data available")
            return

        # Calculate normalized prices
        norm_token1 = self.token1.close_prices / self.token1.close_prices.iloc[0]
        norm_token2 = self.token2.close_prices / self.token2.close_prices.iloc[0]

        # Create figure with 3 subplots
        fig, axs = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle(f"Statistical Arbitrage Analysis: {self.token1.symbol} vs {self.token2.symbol}")

        # Plot 1: Normalized Prices
        axs[0].plot(norm_token1, label=f'{self.token1.symbol} (normalized)')
        axs[0].plot(norm_token2, label=f'{self.token2.symbol} (normalized)')
        axs[0].set_title('Normalized Price Comparison')
        axs[0].legend()
        axs[0].grid(True)

        # Plot 2: Spread
        axs[1].plot(self._spread, label='Spread')
        axs[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axs[1].set_title('Price Spread')
        axs[1].legend()
        axs[1].grid(True)

        # Plot 3: Z-Score
        axs[2].plot(self._zscore, label='Z-Score')
        axs[2].axhline(y=2, color='g', linestyle='--', alpha=0.5, label='Upper Bound')
        axs[2].axhline(y=-2, color='g', linestyle='--', alpha=0.5, label='Lower Bound')
        axs[2].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Mean')
        axs[2].set_title('Z-Score with Trading Bands')
        axs[2].legend()
        axs[2].grid(True)

        # Add cointegration stats to the plot
        stats = self.cointegration_stats
        stats_text = (
            f"Cointegration Stats:\n"
            f"P-value: {stats.get('p_value', 'N/A')}\n"
            f"Hedge Ratio: {stats.get('hedge_ratio', 'N/A')}\n"
            f"Zero Crossings: {stats.get('zero_crossings', 'N/A')}"
        )
        fig.text(0.02, 0.02, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show()

    def backtest(self, entry_std: float = 1.0, exit_std: float = 0.0, initial_capital: float = 1000.0) -> Dict:
        """
        Run a backtest on the cointegrated pair
        
        Args:
            entry_std: Number of standard deviations for entry signal (default: 1.0)
            exit_std: Number of standard deviations for exit signal (default: 0.0 for zero crossing)
            initial_capital: Starting capital in USD (default: 1000.0)
        
        Returns:
            Dictionary containing backtest results
        """
        if not self.is_cointegrated or self._zscore is None or self._spread is None:
            return {
                'success': False,
                'error': 'Pair is not cointegrated or missing data'
            }

        trades: List[Trade] = []
        current_position = None
        entry_price = None
        entry_time = None
        entry_zscore = None
        current_capital = initial_capital

        # Create a DataFrame with all necessary data
        df = pd.DataFrame({
            'zscore': self._zscore,
            'spread': self._spread,
            'token1_price': self.token1.close_prices,
            'token2_price': self.token2.close_prices
        }).dropna()

        for i in range(1, len(df)):
            current_zscore = df['zscore'].iloc[i]
            prev_zscore = df['zscore'].iloc[i-1]
            timestamp = df.index[i]
            
            # Check for zero crossing (exit signal)
            if current_position is not None and (prev_zscore * current_zscore) < 0:
                # Calculate PnL
                exit_price = df['spread'].iloc[i]
                pnl = (exit_price - entry_price) if current_position == 'long' else (entry_price - exit_price)
                
                # Update capital
                current_capital += pnl
                
                # Record trade
                trades.append(Trade(
                    entry_time=entry_time,
                    exit_time=timestamp,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    position=current_position,
                    pnl=pnl,
                    entry_zscore=entry_zscore,
                    exit_zscore=current_zscore
                ))
                
                # Reset position
                current_position = None
                entry_price = None
                entry_time = None
                entry_zscore = None
            
            # Check for entry signals if no position
            elif current_position is None:
                if current_zscore <= -entry_std:
                    # Long signal
                    current_position = 'long'
                    entry_price = df['spread'].iloc[i]
                    entry_time = timestamp
                    entry_zscore = current_zscore
                elif current_zscore >= entry_std:
                    # Short signal
                    current_position = 'short'
                    entry_price = df['spread'].iloc[i]
                    entry_time = timestamp
                    entry_zscore = current_zscore

        # Calculate performance metrics
        if not trades:
            return {
                'success': False,
                'error': 'No trades executed'
            }

        # Convert trades to DataFrame for analysis
        trades_df = pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'position': t.position,
            'pnl': t.pnl,
            'entry_zscore': t.entry_zscore,
            'exit_zscore': t.exit_zscore
        } for t in trades])

        # Calculate metrics
        total_trades = len(trades)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        max_drawdown = self._calculate_max_drawdown(trades_df['pnl'].cumsum())
        
        # Calculate Sharpe Ratio (assuming daily returns)
        returns = trades_df['pnl'].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std()) if len(returns) > 0 else 0

        return {
            'success': True,
            'initial_capital': initial_capital,
            'final_capital': current_capital,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trades': trades_df.to_dict('records')
        }

    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown from cumulative returns"""
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        return abs(drawdowns.min())

    def visualize_backtest(self, entry_std: float = 1.0, exit_std: float = 0.0, initial_capital: float = 1000.0):
        """Visualize backtest results"""
        results = self.backtest(entry_std, exit_std, initial_capital)
        
        if not results['success']:
            print(f"Backtest failed: {results['error']}")
            return

        # Create figure with 4 subplots
        fig, axs = plt.subplots(4, 1, figsize=(16, 16))
        fig.suptitle(f"Backtest Results: {self.token1.symbol} vs {self.token2.symbol}")

        # Plot 1: Z-Score with entry/exit points
        axs[0].plot(self._zscore, label='Z-Score')
        axs[0].axhline(y=entry_std, color='g', linestyle='--', alpha=0.5, label='Upper Entry')
        axs[0].axhline(y=-entry_std, color='g', linestyle='--', alpha=0.5, label='Lower Entry')
        axs[0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Exit')
        
        # Plot entry/exit points
        trades_df = pd.DataFrame(results['trades'])
        for _, trade in trades_df.iterrows():
            # Plot entry point
            axs[0].scatter(trade['entry_time'], trade['entry_zscore'], 
                          marker='^' if trade['position'] == 'long' else 'v',
                          color='g' if trade['position'] == 'long' else 'r',
                          s=100)
            # Plot exit point
            axs[0].scatter(trade['exit_time'], trade['exit_zscore'],
                          marker='o', color='k', s=50)
        
        axs[0].set_title('Z-Score with Entry/Exit Points')
        axs[0].legend()
        axs[0].grid(True)

        # Plot 2: Cumulative PnL
        cumulative_pnl = trades_df['pnl'].cumsum() + results['initial_capital']
        axs[1].plot(cumulative_pnl, label='Portfolio Value')
        axs[1].set_title('Portfolio Value Over Time')
        axs[1].legend()
        axs[1].grid(True)

        # Plot 3: Trade Distribution
        axs[2].hist(trades_df['pnl'], bins=50, label='PnL Distribution')
        axs[2].set_title('Trade PnL Distribution')
        axs[2].legend()
        axs[2].grid(True)

        # Plot 4: Drawdown
        drawdown = self._calculate_max_drawdown(cumulative_pnl)
        axs[3].plot(cumulative_pnl / cumulative_pnl.expanding().max() - 1, label='Drawdown')
        axs[3].set_title(f'Drawdown (Max: {drawdown:.2%})')
        axs[3].legend()
        axs[3].grid(True)

        # Add performance metrics to the plot
        metrics_text = (
            f"Performance Metrics:\n"
            f"Initial Capital: ${results['initial_capital']:.2f}\n"
            f"Final Capital: ${results['final_capital']:.2f}\n"
            f"Total Return: {((results['final_capital']/results['initial_capital'])-1):.2%}\n"
            f"Total Trades: {results['total_trades']}\n"
            f"Win Rate: {results['win_rate']:.2%}\n"
            f"Total PnL: ${results['total_pnl']:.2f}\n"
            f"Avg PnL: ${results['avg_pnl']:.2f}\n"
            f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {results['max_drawdown']:.2%}"
        )
        fig.text(0.02, 0.02, metrics_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show()
