"""
Pairs Trading Strategy Module
Implements statistical arbitrage based on cointegration
"""
import base64
import io
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint


class PairsTradingError(Exception):
    """Base exception for pairs trading errors"""
    pass


class InvalidDateRangeError(PairsTradingError):
    """Raised when date range is invalid"""
    pass


class InsufficientDataError(PairsTradingError):
    """Raised when insufficient data is available"""
    pass


class CointegrationRequirementError(PairsTradingError):
    """Raised when cointegration test fails"""
    pass


@dataclass
class PairsTradingResult:
    """Results from pairs trading analysis"""
    stock1: str
    stock2: str
    start_date: str
    end_date: str
    cointegration_score: float
    cointegration_p_value: float
    cointegration_passed: bool
    spread_plot_base64: str
    pnl_plot_base64: str
    trading_signals: pd.DataFrame
    summary_stats: dict


def run_pairs_analysis(
    stock1: str,
    stock2: str,
    start_date: str,
    end_date: str,
    run_cointegration_test: bool = False
) -> PairsTradingResult:
    """
    Run pairs trading analysis on two stocks
    
    Args:
        stock1: First stock ticker
        stock2: Second stock ticker
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        run_cointegration_test: Whether to show detailed cointegration results
    
    Returns:
        PairsTradingResult object
    """
    # Validate inputs
    stock1 = stock1.strip().upper()
    stock2 = stock2.strip().upper()
    
    if not stock1 or not stock2:
        raise PairsTradingError("Stock tickers cannot be empty")
    
    # Validate dates
    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
    except Exception as e:
        raise InvalidDateRangeError(f"Invalid date format: {e}")
    
    if start_dt >= end_dt:
        raise InvalidDateRangeError("Start date must be before end date")
    
    if end_dt > pd.Timestamp.now():
        raise InvalidDateRangeError("End date cannot be in the future")
    
    # Download data
    try:
        data1 = yf.download(stock1, start=start_date, end=end_date, progress=False)
        data2 = yf.download(stock2, start=start_date, end=end_date, progress=False)
    except Exception as e:
        raise PairsTradingError(f"Failed to download stock data: {e}")
    
    if data1.empty or data2.empty:
        raise InsufficientDataError("No data available for the given date range")
    
    if len(data1) < 30 or len(data2) < 30:
        raise InsufficientDataError("Insufficient data (minimum 30 days required)")
    
    # Align data
    df = pd.DataFrame({
        stock1: data1['Close'],
        stock2: data2['Close']
    }).dropna()
    
    if len(df) < 30:
        raise InsufficientDataError("Insufficient overlapping data")
    
    # Cointegration test
    score, p_value, _ = coint(df[stock1], df[stock2])
    cointegration_passed = p_value < 0.05
    
    # Calculate spread
    # Simple spread: Stock1 - beta * Stock2
    beta = np.polyfit(df[stock2], df[stock1], 1)[0]
    df['spread'] = df[stock1] - beta * df[stock2]
    
    # Calculate trading signals
    mean_spread = df['spread'].mean()
    std_spread = df['spread'].std()
    
    df['z_score'] = (df['spread'] - mean_spread) / std_spread
    df['upper_threshold'] = mean_spread + 2 * std_spread
    df['lower_threshold'] = mean_spread - 2 * std_spread
    
    # Generate signals
    df['signal'] = 0
    df.loc[df['z_score'] > 2, 'signal'] = -1  # Short spread
    df.loc[df['z_score'] < -2, 'signal'] = 1   # Long spread
    
    # Calculate PnL
    df['position'] = df['signal'].shift(1).fillna(0)
    df['returns'] = df['spread'].pct_change()
    df['strategy_returns'] = df['position'] * df['returns']
    df['cumulative_pnl'] = df['strategy_returns'].cumsum()
    
    # Summary stats
    num_trades = (df['signal'] != 0).sum()
    num_long = (df['signal'] == 1).sum()
    num_short = (df['signal'] == -1).sum()
    total_pnl = df['cumulative_pnl'].iloc[-1] if len(df) > 0 else 0
    
    summary_stats = {
        'trading_score': score,
        'trading_p_value': p_value,
        'trading_cointegration_passed': cointegration_passed,
        'mean_spread': mean_spread,
        'std_spread': std_spread,
        'upper_threshold': mean_spread + 2 * std_spread,
        'lower_threshold': mean_spread - 2 * std_spread,
        'num_trades': num_trades,
        'num_long': num_long,
        'num_short': num_short,
        'total_pnl': total_pnl
    }
    
    # Create plots
    spread_plot = _create_spread_plot(df, stock1, stock2, mean_spread, std_spread)
    pnl_plot = _create_pnl_plot(df)
    
    # Recent signals (last 30 days)
    recent_signals = df[df['signal'] != 0].tail(30).copy()
    recent_signals = recent_signals[['spread', 'z_score', 'signal']].reset_index()
    
    return PairsTradingResult(
        stock1=stock1,
        stock2=stock2,
        start_date=start_date,
        end_date=end_date,
        cointegration_score=score,
        cointegration_p_value=p_value,
        cointegration_passed=cointegration_passed,
        spread_plot_base64=spread_plot,
        pnl_plot_base64=pnl_plot,
        trading_signals=recent_signals,
        summary_stats=summary_stats
    )


def _create_spread_plot(df: pd.DataFrame, stock1: str, stock2: str, mean: float, std: float) -> str:
    """Create spread analysis plot"""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0a0b11')
    ax.set_facecolor('#0a0b11')
    
    ax.plot(df.index, df['spread'], label='Spread', color='#d9b36a', linewidth=1.5)
    ax.axhline(mean, color='#5e9c7e', linestyle='--', label='Mean', linewidth=1.5)
    ax.axhline(mean + 2*std, color='#e63946', linestyle='--', label='Upper Threshold', linewidth=1.5)
    ax.axhline(mean - 2*std, color='#e63946', linestyle='--', label='Lower Threshold', linewidth=1.5)
    
    ax.set_title(f'Spread Analysis: {stock1} vs {stock2}', color='#f8fafc', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', color='#f8fafc')
    ax.set_ylabel('Spread', color='#f8fafc')
    ax.tick_params(colors='#f8fafc')
    ax.legend(facecolor='#0a0b11', edgecolor='#f8fafc', labelcolor='#f8fafc')
    ax.grid(True, alpha=0.2, color='#f8fafc')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#0a0b11', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


def _create_pnl_plot(df: pd.DataFrame) -> str:
    """Create cumulative PnL plot"""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0a0b11')
    ax.set_facecolor('#0a0b11')
    
    ax.plot(df.index, df['cumulative_pnl'], label='Cumulative PnL', color='#5e9c7e', linewidth=2)
    ax.axhline(0, color='#f8fafc', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_title('Cumulative Strategy Returns', color='#f8fafc', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', color='#f8fafc')
    ax.set_ylabel('Cumulative PnL', color='#f8fafc')
    ax.tick_params(colors='#f8fafc')
    ax.legend(facecolor='#0a0b11', edgecolor='#f8fafc', labelcolor='#f8fafc')
    ax.grid(True, alpha=0.2, color='#f8fafc')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#0a0b11', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64
