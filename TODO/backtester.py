import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from dataclasses import dataclass
from .coint_pair import CointPair

@dataclass
class Trade:
    """Class to store trade information"""
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    position: str  # 'long' or 'short'
    pnl: Optional[float]
    entry_zscore: float
    exit_zscore: Optional[float]

@dataclass
class BacktestResult:
    """Class to store backtest results"""
    success: bool
    error: Optional[str]
    initial_capital: float
    final_capital: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    trades: List[Dict]

class Backtester:
    def __init__(self, pair: CointPair):
        """Initialize backtester with a cointegrated pair"""
        self.pair = pair
        self._trades: List[Trade] = []
        self._current_position = None
        self._entry_price = None
        self._entry_time = None
        self._entry_zscore = None
        self._current_capital = 0.0

    def run(self, entry_std: float = 1.0, exit_std: float = 0.0, initial_capital: float = 1000.0) -> BacktestResult:
        """
        Run backtest on the pair
        
        Args:
            entry_std: Number of standard deviations for entry signal (default: 1.0)
            exit_std: Number of standard deviations for exit signal (default: 0.0 for zero crossing)
            initial_capital: Starting capital in USD (default: 1000.0)
        
        Returns:
            BacktestResult object containing all backtest results
        """
        if not self.pair.is_cointegrated or self.pair.zscore is None or self.pair.spread is None:
            return BacktestResult(
                success=False,
                error='Pair is not cointegrated or missing data',
                initial_capital=initial_capital,
                final_capital=initial_capital,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                avg_pnl=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                trades=[]
            )

        self._trades = []
        self._current_position = None
        self._entry_price = None
        self._entry_time = None
        self._entry_zscore = None
        self._current_capital = initial_capital

        # Create a DataFrame with all necessary data
        df = pd.DataFrame({
            'zscore': self.pair.zscore,
            'spread': self.pair.spread,
            'token1_price': self.pair.token1.close_prices,
            'token2_price': self.pair.token2.close_prices
        }).dropna()

        for i in range(1, len(df)):
            current_zscore = df['zscore'].iloc[i]
            prev_zscore = df['zscore'].iloc[i-1]
            timestamp = df.index[i]
            
            # Check for zero crossing (exit signal)
            if self._current_position is not None and (prev_zscore * current_zscore) < 0:
                # Calculate PnL
                exit_price = df['spread'].iloc[i]
                pnl = (exit_price - self._entry_price) if self._current_position == 'long' else (self._entry_price - exit_price)
                
                # Update capital
                self._current_capital += pnl
                
                # Record trade
                self._trades.append(Trade(
                    entry_time=self._entry_time,
                    exit_time=timestamp,
                    entry_price=self._entry_price,
                    exit_price=exit_price,
                    position=self._current_position,
                    pnl=pnl,
                    entry_zscore=self._entry_zscore,
                    exit_zscore=current_zscore
                ))
                
                # Reset position
                self._current_position = None
                self._entry_price = None
                self._entry_time = None
                self._entry_zscore = None
            
            # Check for entry signals if no position
            elif self._current_position is None:
                if current_zscore <= -entry_std:
                    # Long signal
                    self._current_position = 'long'
                    self._entry_price = df['spread'].iloc[i]
                    self._entry_time = timestamp
                    self._entry_zscore = current_zscore
                elif current_zscore >= entry_std:
                    # Short signal
                    self._current_position = 'short'
                    self._entry_price = df['spread'].iloc[i]
                    self._entry_time = timestamp
                    self._entry_zscore = current_zscore

        # Calculate performance metrics
        if not self._trades:
            return BacktestResult(
                success=False,
                error='No trades executed',
                initial_capital=initial_capital,
                final_capital=initial_capital,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                avg_pnl=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                trades=[]
            )

        # Convert trades to DataFrame for analysis
        trades_df = pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'position': t.position,
            'pnl': t.pnl,
            'entry_zscore': t.entry_zscore,
            'exit_zscore': t.exit_zscore
        } for t in self._trades])

        # Calculate metrics
        total_trades = len(self._trades)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        max_drawdown = self._calculate_max_drawdown(trades_df['pnl'].cumsum())
        
        # Calculate Sharpe Ratio (assuming daily returns)
        returns = trades_df['pnl'].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std()) if len(returns) > 0 else 0

        return BacktestResult(
            success=True,
            error=None,
            initial_capital=initial_capital,
            final_capital=self._current_capital,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_pnl=avg_pnl,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            trades=trades_df.to_dict('records')
        )

    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown from cumulative returns"""
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        return abs(drawdowns.min())

    def visualize(self, entry_std: float = 1.0, exit_std: float = 0.0, initial_capital: float = 1000.0):
        """Visualize backtest results"""
        results = self.run(entry_std, exit_std, initial_capital)
        
        if not results.success:
            print(f"Backtest failed: {results.error}")
            return

        # Create figure with 4 subplots
        fig, axs = plt.subplots(4, 1, figsize=(16, 16))
        fig.suptitle(f"Backtest Results: {self.pair.token1.symbol} vs {self.pair.token2.symbol}")

        # Plot 1: Z-Score with entry/exit points
        axs[0].plot(self.pair.zscore, label='Z-Score')
        axs[0].axhline(y=entry_std, color='g', linestyle='--', alpha=0.5, label='Upper Entry')
        axs[0].axhline(y=-entry_std, color='g', linestyle='--', alpha=0.5, label='Lower Entry')
        axs[0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Exit')
        
        # Plot entry/exit points
        trades_df = pd.DataFrame(results.trades)
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
        cumulative_pnl = trades_df['pnl'].cumsum() + results.initial_capital
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
            f"Initial Capital: ${results.initial_capital:.2f}\n"
            f"Final Capital: ${results.final_capital:.2f}\n"
            f"Total Return: {((results.final_capital/results.initial_capital)-1):.2%}\n"
            f"Total Trades: {results.total_trades}\n"
            f"Win Rate: {results.win_rate:.2%}\n"
            f"Total PnL: ${results.total_pnl:.2f}\n"
            f"Avg PnL: ${results.avg_pnl:.2f}\n"
            f"Sharpe Ratio: {results.sharpe_ratio:.2f}\n"
            f"Max Drawdown: {results.max_drawdown:.2%}"
        )
        fig.text(0.02, 0.02, metrics_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show() 