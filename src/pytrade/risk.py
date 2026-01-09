"""
Technical Analysis Features for Stocks and Crypto Trading.

This module provides a comprehensive set of technical indicators commonly used
in quantitative trading strategies for both stocks and cryptocurrencies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple
from collections import deque


#  trend indicators

def sma(prices: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average (SMA).
    
    Args:
        prices: Series of prices (typically close prices)
        period: Number of periods for the moving average
    
    Returns:
        Series with SMA values
    """
    return prices.rolling(window=period).mean()


def ema(prices: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average (EMA).
    
    Gives more weight to recent prices, making it more responsive
    to new information compared to SMA.
    
    Args:
        prices: Series of prices
        period: Number of periods for the moving average
    
    Returns:
        Series with EMA values
    """
    return prices.ewm(span=period, adjust=False).mean()


def wma(prices: pd.Series, period: int) -> pd.Series:
    """
    Weighted Moving Average (WMA).
    
    Assigns linearly increasing weights to more recent prices.
    
    Args:
        prices: Series of prices
        period: Number of periods
    
    Returns:
        Series with WMA values
    """
    weights = np.arange(1, period + 1)
    return prices.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence (MACD).
    
    A trend-following momentum indicator showing the relationship
    between two EMAs of prices.
    
    Args:
        prices: Series of prices
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
    
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average Directional Index (ADX).
    
    Measures trend strength regardless of direction.
    ADX > 25 indicates a strong trend.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        period: Lookback period (default 14)
    
    Returns:
        Series with ADX values
    """
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_values = dx.rolling(window=period).mean()
    
    return adx_values


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series, 
               period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """
    SuperTrend Indicator.
    
    A trend-following indicator popular in crypto trading.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        period: ATR period (default 10)
        multiplier: ATR multiplier (default 3.0)
    
    Returns:
        Tuple of (SuperTrend values, Direction: 1 for uptrend, -1 for downtrend)
    """
    atr_val = atr(high, low, close, period)
    hl2 = (high + low) / 2
    
    upper_band = hl2 + (multiplier * atr_val)
    lower_band = hl2 - (multiplier * atr_val)
    
    supertrend_vals = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=float)
    
    for i in range(period, len(close)):
        if close.iloc[i] > upper_band.iloc[i-1]:
            supertrend_vals.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i-1]:
            supertrend_vals.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
        else:
            if direction.iloc[i-1] == 1:
                supertrend_vals.iloc[i] = max(lower_band.iloc[i], supertrend_vals.iloc[i-1])
                direction.iloc[i] = 1
            else:
                supertrend_vals.iloc[i] = min(upper_band.iloc[i], supertrend_vals.iloc[i-1])
                direction.iloc[i] = -1
    
    return supertrend_vals, direction


# momentum indicators

def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI).
    
    Measures the speed and magnitude of price changes.
    RSI > 70 indicates overbought, RSI < 30 indicates oversold.
    
    Args:
        prices: Series of prices
        period: Lookback period (default 14)
    
    Returns:
        Series with RSI values (0-100)
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))
    
    return rsi_values


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator.
    
    Compares closing price to price range over a period.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        k_period: %K period (default 14)
        d_period: %D period (default 3)
    
    Returns:
        Tuple of (%K, %D)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(window=d_period).mean()
    
    return stoch_k, stoch_d


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Williams %R.
    
    Momentum indicator measuring overbought/oversold levels.
    Similar to stochastic but inverted scale.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        period: Lookback period (default 14)
    
    Returns:
        Series with Williams %R values (-100 to 0)
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """
    Commodity Channel Index (CCI).
    
    Measures current price level relative to average price.
    Popular in crypto trading for identifying cyclical trends.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        period: Lookback period (default 20)
    
    Returns:
        Series with CCI values
    """
    tp = (high + low + close) / 3  # Typical Price
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    
    cci_values = (tp - sma_tp) / (0.015 * mad)
    return cci_values


def roc(prices: pd.Series, period: int = 12) -> pd.Series:
    """
    Rate of Change (ROC).
    
    Measures percentage change between current price and price n periods ago.
    
    Args:
        prices: Series of prices
        period: Lookback period (default 12)
    
    Returns:
        Series with ROC values (percentage)
    """
    return ((prices - prices.shift(period)) / prices.shift(period)) * 100


def momentum(prices: pd.Series, period: int = 10) -> pd.Series:
    """
    Momentum Indicator.
    
    Simple price difference over n periods.
    
    Args:
        prices: Series of prices
        period: Lookback period (default 10)
    
    Returns:
        Series with momentum values
    """
    return prices - prices.shift(period)


# volatility indicators

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR).
    
    Measures market volatility by calculating the average range.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        period: Lookback period (default 14)
    
    Returns:
        Series with ATR values
    """
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return tr.rolling(window=period).mean()


def bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.
    
    Volatility bands placed above and below a moving average.
    
    Args:
        prices: Series of prices
        period: Moving average period (default 20)
        std_dev: Number of standard deviations (default 2.0)
    
    Returns:
        Tuple of (Upper Band, Middle Band, Lower Band)
    """
    middle = sma(prices, period)
    std = prices.rolling(window=period).std()
    
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    
    return upper, middle, lower


def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                     ema_period: int = 20, atr_period: int = 10, 
                     multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Keltner Channels.
    
    Volatility-based envelopes using ATR instead of standard deviation.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        ema_period: EMA period for middle line (default 20)
        atr_period: ATR period (default 10)
        multiplier: ATR multiplier (default 2.0)
    
    Returns:
        Tuple of (Upper Channel, Middle Channel, Lower Channel)
    """
    middle = ema(close, ema_period)
    atr_val = atr(high, low, close, atr_period)
    
    upper = middle + (multiplier * atr_val)
    lower = middle - (multiplier * atr_val)
    
    return upper, middle, lower


def donchian_channels(high: pd.Series, low: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Donchian Channels.
    
    Price channels based on highest high and lowest low.
    Popular for breakout strategies.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        period: Lookback period (default 20)
    
    Returns:
        Tuple of (Upper Channel, Middle Channel, Lower Channel)
    """
    upper = high.rolling(window=period).max()
    lower = low.rolling(window=period).min()
    middle = (upper + lower) / 2
    
    return upper, middle, lower


def historical_volatility(prices: pd.Series, period: int = 20, annualize: bool = True) -> pd.Series:
    """
    Historical Volatility (HV).
    
    Standard deviation of log returns, optionally annualized.
    
    Args:
        prices: Series of prices
        period: Lookback period (default 20)
        annualize: Whether to annualize (default True, assumes 252 trading days)
    
    Returns:
        Series with volatility values
    """
    log_returns = np.log(prices / prices.shift(1))
    vol = log_returns.rolling(window=period).std()
    
    if annualize:
        vol = vol * np.sqrt(252)
    
    return vol


# volume indicators

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume (OBV).
    
    Cumulative volume indicator that adds volume on up days
    and subtracts on down days.
    
    Args:
        close: Series of close prices
        volume: Series of volume
    
    Returns:
        Series with OBV values
    """
    direction = np.sign(close.diff())
    return (direction * volume).cumsum()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Volume Weighted Average Price (VWAP).
    
    Average price weighted by volume, commonly used as a benchmark.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        volume: Series of volume
    
    Returns:
        Series with VWAP values
    """
    tp = (high + low + close) / 3
    return (tp * volume).cumsum() / volume.cumsum()


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, 
        volume: pd.Series, period: int = 14) -> pd.Series:
    """
    Money Flow Index (MFI).
    
    Volume-weighted RSI, measures buying and selling pressure.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        volume: Series of volume
        period: Lookback period (default 14)
    
    Returns:
        Series with MFI values (0-100)
    """
    tp = (high + low + close) / 3
    raw_money_flow = tp * volume
    
    positive_flow = np.where(tp > tp.shift(1), raw_money_flow, 0)
    negative_flow = np.where(tp < tp.shift(1), raw_money_flow, 0)
    
    positive_mf = pd.Series(positive_flow).rolling(window=period).sum()
    negative_mf = pd.Series(negative_flow).rolling(window=period).sum()
    
    mfi_values = 100 - (100 / (1 + positive_mf / negative_mf))
    return mfi_values


def accumulation_distribution(high: pd.Series, low: pd.Series, 
                               close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Accumulation/Distribution Line (A/D).
    
    Measures the cumulative flow of money into and out of a security.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        volume: Series of volume
    
    Returns:
        Series with A/D values
    """
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.fillna(0)
    ad = (clv * volume).cumsum()
    return ad


def chaikin_money_flow(high: pd.Series, low: pd.Series, close: pd.Series,
                       volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Chaikin Money Flow (CMF).
    
    Measures buying and selling pressure over a period.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        volume: Series of volume
        period: Lookback period (default 20)
    
    Returns:
        Series with CMF values (-1 to 1)
    """
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.fillna(0)
    
    cmf = (clv * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
    return cmf


# suppert/resistance indicators

def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
    """
    Standard Pivot Points.
    
    Calculate support and resistance levels based on previous period.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
    
    Returns:
        Dictionary with pivot point levels
    """
    pivot = (high + low + close) / 3
    
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    
    return {
        'pivot': pivot,
        'r1': r1, 'r2': r2, 'r3': r3,
        's1': s1, 's2': s2, 's3': s3
    }


def fibonacci_retracement(high_price: float, low_price: float) -> dict:
    """
    Fibonacci Retracement Levels.
    
    Calculate key Fibonacci levels between a high and low.
    
    Args:
        high_price: The high price point
        low_price: The low price point
    
    Returns:
        Dictionary with Fibonacci levels
    """
    diff = high_price - low_price
    
    return {
        'level_0': high_price,
        'level_236': high_price - 0.236 * diff,
        'level_382': high_price - 0.382 * diff,
        'level_500': high_price - 0.500 * diff,
        'level_618': high_price - 0.618 * diff,
        'level_786': high_price - 0.786 * diff,
        'level_100': low_price
    }


# crypto indicators 

def fear_greed_oscillator(prices: pd.Series, volume: pd.Series, 
                          period: int = 14) -> pd.Series:
    """
    Fear & Greed Oscillator.
    
    Custom indicator combining momentum and volume to gauge market sentiment.
    Higher values indicate greed, lower values indicate fear.
    
    Args:
        prices: Series of prices
        volume: Series of volume
        period: Lookback period (default 14)
    
    Returns:
        Series with oscillator values (0-100)
    """
    # Combine RSI and volume momentum
    rsi_val = rsi(prices, period)
    vol_sma = sma(volume, period)
    vol_ratio = (volume / vol_sma) * 50
    
    # Weighted combination
    oscillator = 0.6 * rsi_val + 0.4 * vol_ratio.clip(0, 100)
    return oscillator


def nvt_ratio(price: pd.Series, network_value: pd.Series, 
              transaction_volume: pd.Series) -> pd.Series:
    """
    Network Value to Transaction Ratio (NVT).
    
    Crypto-specific indicator comparing market cap to transaction volume.
    Often called "crypto's P/E ratio".
    
    Args:
        price: Series of prices
        network_value: Series of network/market cap values
        transaction_volume: Series of on-chain transaction volumes
    
    Returns:
        Series with NVT ratio values
    """
    return network_value / transaction_volume


# dataclass for technical indicators snapshot

@dataclass
class TechnicalSnapshot:
    """Snapshot of all technical indicators at a point in time."""
    
    # Trend
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    
    # Momentum
    rsi_14: Optional[float] = None
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    williams_r: Optional[float] = None
    cci: Optional[float] = None
    
    # Volatility
    atr_14: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    historical_vol: Optional[float] = None
    
    # Volume
    obv: Optional[float] = None
    vwap: Optional[float] = None
    mfi: Optional[float] = None
    
    # Signals
    trend_direction: Optional[str] = None  # "bullish", "bearish", "neutral"
    momentum_signal: Optional[str] = None  # "overbought", "oversold", "neutral"


class TechnicalAnalyzer:
    """
    Complete technical analysis engine for stocks and crypto.
    
    Computes all indicators from OHLCV data and provides signals.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV DataFrame.
        
        Args:
            df: DataFrame with columns: 'open', 'high', 'low', 'close', 'volume'
        """
        self.df = df.copy()
        self._validate_data()
        self._compute_all_indicators()
    
    def _validate_data(self) -> None:
        """Validate input DataFrame has required columns."""
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def _compute_all_indicators(self) -> None:
        """Compute all technical indicators."""
        h, l, c, v = self.df['high'], self.df['low'], self.df['close'], self.df['volume']
        
        # Trend Indicators
        self.df['sma_20'] = sma(c, 20)
        self.df['sma_50'] = sma(c, 50)
        self.df['sma_200'] = sma(c, 200)
        self.df['ema_12'] = ema(c, 12)
        self.df['ema_26'] = ema(c, 26)
        
        macd_line, macd_signal, macd_hist = macd(c)
        self.df['macd_line'] = macd_line
        self.df['macd_signal'] = macd_signal
        self.df['macd_histogram'] = macd_hist
        
        # Momentum Indicators
        self.df['rsi_14'] = rsi(c, 14)
        stoch_k, stoch_d = stochastic(h, l, c)
        self.df['stoch_k'] = stoch_k
        self.df['stoch_d'] = stoch_d
        self.df['williams_r'] = williams_r(h, l, c)
        self.df['cci'] = cci(h, l, c)
        
        # Volatility Indicators
        self.df['atr_14'] = atr(h, l, c, 14)
        bb_upper, bb_middle, bb_lower = bollinger_bands(c)
        self.df['bb_upper'] = bb_upper
        self.df['bb_middle'] = bb_middle
        self.df['bb_lower'] = bb_lower
        self.df['historical_vol'] = historical_volatility(c)
        
        # Volume Indicators
        self.df['obv'] = obv(c, v)
        self.df['vwap'] = vwap(h, l, c, v)
        self.df['mfi'] = mfi(h, l, c, v)
    
    def get_snapshot(self, index: int = -1) -> TechnicalSnapshot:
        """
        Get technical indicator snapshot at a specific index.
        
        Args:
            index: DataFrame index (default -1 for latest)
        
        Returns:
            TechnicalSnapshot with all indicator values
        """
        row = self.df.iloc[index]
        
        # Determine trend direction
        trend = "neutral"
        if pd.notna(row.get('sma_50')) and pd.notna(row.get('sma_200')):
            if row['sma_50'] > row['sma_200']:
                trend = "bullish"
            elif row['sma_50'] < row['sma_200']:
                trend = "bearish"
        
        # Determine momentum signal
        momentum_sig = "neutral"
        if pd.notna(row.get('rsi_14')):
            if row['rsi_14'] > 70:
                momentum_sig = "overbought"
            elif row['rsi_14'] < 30:
                momentum_sig = "oversold"
        
        return TechnicalSnapshot(
            sma_20=row.get('sma_20'),
            sma_50=row.get('sma_50'),
            sma_200=row.get('sma_200'),
            ema_12=row.get('ema_12'),
            ema_26=row.get('ema_26'),
            macd_line=row.get('macd_line'),
            macd_signal=row.get('macd_signal'),
            macd_histogram=row.get('macd_histogram'),
            rsi_14=row.get('rsi_14'),
            stoch_k=row.get('stoch_k'),
            stoch_d=row.get('stoch_d'),
            williams_r=row.get('williams_r'),
            cci=row.get('cci'),
            atr_14=row.get('atr_14'),
            bb_upper=row.get('bb_upper'),
            bb_middle=row.get('bb_middle'),
            bb_lower=row.get('bb_lower'),
            historical_vol=row.get('historical_vol'),
            obv=row.get('obv'),
            vwap=row.get('vwap'),
            mfi=row.get('mfi'),
            trend_direction=trend,
            momentum_signal=momentum_sig
        )
    
    def get_dataframe(self) -> pd.DataFrame:
        """Return DataFrame with all computed indicators."""
        return self.df


# value at risk (var) calculations

def calculate_returns(prices: pd.Series, method: str = "simple") -> pd.Series:
    """
    Calculate returns from price series.
    
    Args:
        prices: Series of prices
        method: "simple" for arithmetic returns, "log" for logarithmic returns
    
    Returns:
        Series of returns
    """
    if method == "log":
        return np.log(prices / prices.shift(1))
    return prices.pct_change()


def var_historical(returns: pd.Series, confidence_level: float = 0.95, 
                   portfolio_value: float = 1.0) -> float:
    """
    Historical Value at Risk (VaR).
    
    Non-parametric VaR using historical simulation method.
    
    Args:
        returns: Series of historical returns
        confidence_level: Confidence level (default 0.95 for 95%)
        portfolio_value: Current portfolio value (default 1.0 for percentage)
    
    Returns:
        VaR as a positive number (potential loss)
    """
    returns_clean = returns.dropna()
    if len(returns_clean) == 0:
        return np.nan
    
    percentile = (1 - confidence_level) * 100
    var = -np.percentile(returns_clean, percentile) * portfolio_value
    return var


def var_parametric(returns: pd.Series, confidence_level: float = 0.95,
                   portfolio_value: float = 1.0) -> float:
    """
    Parametric (Variance-Covariance) Value at Risk.
    
    Assumes returns are normally distributed.
    
    Args:
        returns: Series of historical returns
        confidence_level: Confidence level (default 0.95)
        portfolio_value: Current portfolio value
    
    Returns:
        VaR as a positive number (potential loss)
    """
    from scipy import stats
    
    returns_clean = returns.dropna()
    if len(returns_clean) == 0:
        return np.nan
    
    mean = returns_clean.mean()
    std = returns_clean.std()
    z_score = stats.norm.ppf(1 - confidence_level)
    
    var = -(mean + z_score * std) * portfolio_value
    return var


def var_monte_carlo(returns: pd.Series, confidence_level: float = 0.95,
                    portfolio_value: float = 1.0, simulations: int = 10000,
                    time_horizon: int = 1) -> float:
    """
    Monte Carlo Value at Risk.
    
    Simulates future returns using random sampling.
    
    Args:
        returns: Series of historical returns
        confidence_level: Confidence level (default 0.95)
        portfolio_value: Current portfolio value
        simulations: Number of Monte Carlo simulations
        time_horizon: Number of periods to simulate
    
    Returns:
        VaR as a positive number (potential loss)
    """
    returns_clean = returns.dropna()
    if len(returns_clean) == 0:
        return np.nan
    
    mean = returns_clean.mean()
    std = returns_clean.std()
    
    # Simulate returns
    simulated_returns = np.random.normal(mean, std, (simulations, time_horizon))
    cumulative_returns = np.prod(1 + simulated_returns, axis=1) - 1
    
    percentile = (1 - confidence_level) * 100
    var = -np.percentile(cumulative_returns, percentile) * portfolio_value
    return var


def var_cornish_fisher(returns: pd.Series, confidence_level: float = 0.95,
                       portfolio_value: float = 1.0) -> float:
    """
    Cornish-Fisher Value at Risk.
    
    Adjusts for skewness and kurtosis in the return distribution.
    More accurate for non-normal distributions (common in crypto).
    
    Args:
        returns: Series of historical returns
        confidence_level: Confidence level
        portfolio_value: Current portfolio value
    
    Returns:
        VaR as a positive number (potential loss)
    """
    from scipy import stats
    
    returns_clean = returns.dropna()
    if len(returns_clean) < 4:
        return np.nan
    
    mean = returns_clean.mean()
    std = returns_clean.std()
    skew = returns_clean.skew()
    kurt = returns_clean.kurtosis()
    
    z = stats.norm.ppf(1 - confidence_level)
    
    # Cornish-Fisher expansion
    z_cf = (z + (z**2 - 1) * skew / 6 +
            (z**3 - 3*z) * (kurt - 3) / 24 -
            (2*z**3 - 5*z) * (skew**2) / 36)
    
    var = -(mean + z_cf * std) * portfolio_value
    return var


def rolling_var(returns: pd.Series, window: int = 252, 
                confidence_level: float = 0.95) -> pd.Series:
    """
    Rolling Historical VaR.
    
    Computes VaR over a rolling window.
    
    Args:
        returns: Series of returns
        window: Rolling window size
        confidence_level: Confidence level
    
    Returns:
        Series of rolling VaR values
    """
    percentile = (1 - confidence_level) * 100
    return -returns.rolling(window=window).quantile(percentile / 100)


# expected shortfall (es) calculations

def expected_shortfall(returns: pd.Series, confidence_level: float = 0.95,
                       portfolio_value: float = 1.0) -> float:
    """
    Expected Shortfall (ES) / Conditional VaR (CVaR).
    
    Average loss beyond the VaR threshold. More coherent risk measure than VaR.
    
    Args:
        returns: Series of historical returns
        confidence_level: Confidence level (default 0.95)
        portfolio_value: Current portfolio value
    
    Returns:
        ES as a positive number (expected loss in the tail)
    """
    returns_clean = returns.dropna()
    if len(returns_clean) == 0:
        return np.nan
    
    var = var_historical(returns_clean, confidence_level, 1.0)
    # Get returns worse than VaR
    tail_returns = returns_clean[returns_clean <= -var]
    
    if len(tail_returns) == 0:
        return var * portfolio_value
    
    es = -tail_returns.mean() * portfolio_value
    return es


def es_parametric(returns: pd.Series, confidence_level: float = 0.95,
                  portfolio_value: float = 1.0) -> float:
    """
    Parametric Expected Shortfall.
    
    Assumes normal distribution.
    
    Args:
        returns: Series of historical returns
        confidence_level: Confidence level
        portfolio_value: Current portfolio value
    
    Returns:
        ES as a positive number
    """
    from scipy import stats
    
    returns_clean = returns.dropna()
    if len(returns_clean) == 0:
        return np.nan
    
    mean = returns_clean.mean()
    std = returns_clean.std()
    alpha = 1 - confidence_level
    
    # ES for normal distribution
    z = stats.norm.ppf(alpha)
    es = -mean + std * stats.norm.pdf(z) / alpha
    
    return es * portfolio_value


def rolling_es(returns: pd.Series, window: int = 252,
               confidence_level: float = 0.95) -> pd.Series:
    """
    Rolling Expected Shortfall.
    
    Args:
        returns: Series of returns
        window: Rolling window size
        confidence_level: Confidence level
    
    Returns:
        Series of rolling ES values
    """
    def _es(x):
        var = -np.percentile(x, (1 - confidence_level) * 100)
        return -x[x <= -var].mean() if len(x[x <= -var]) > 0 else var
    
    return returns.rolling(window=window).apply(_es, raw=False)


# risk management - drawdown analysis

def drawdown(prices: pd.Series) -> pd.Series:
    """
    Calculate drawdown series.
    
    Drawdown measures the decline from a historical peak.
    
    Args:
        prices: Series of prices or portfolio values
    
    Returns:
        Series of drawdown values (negative numbers)
    """
    cummax = prices.cummax()
    return (prices - cummax) / cummax


def max_drawdown(prices: pd.Series) -> float:
    """
    Maximum Drawdown (MDD).
    
    The largest peak-to-trough decline.
    
    Args:
        prices: Series of prices or portfolio values
    
    Returns:
        Maximum drawdown as a positive percentage
    """
    dd = drawdown(prices)
    return -dd.min()


def drawdown_duration(prices: pd.Series) -> Tuple[int, pd.Series]:
    """
    Drawdown Duration Analysis.
    
    Args:
        prices: Series of prices
    
    Returns:
        Tuple of (max duration in periods, Series of duration at each point)
    """
    dd = drawdown(prices)
    
    # Track duration
    duration = pd.Series(0, index=prices.index)
    current_duration = 0
    
    for i in range(len(dd)):
        if dd.iloc[i] < 0:
            current_duration += 1
        else:
            current_duration = 0
        duration.iloc[i] = current_duration
    
    return duration.max(), duration


def calmar_ratio(returns: pd.Series, prices: pd.Series, 
                 periods_per_year: int = 252) -> float:
    """
    Calmar Ratio.
    
    Annualized return divided by maximum drawdown.
    
    Args:
        returns: Series of returns
        prices: Series of prices for drawdown calculation
        periods_per_year: Trading periods per year (252 for daily)
    
    Returns:
        Calmar ratio
    """
    ann_return = returns.mean() * periods_per_year
    mdd = max_drawdown(prices)
    
    if mdd == 0:
        return np.nan
    
    return ann_return / mdd


def ulcer_index(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Ulcer Index.
    
    Measures downside volatility and drawdown depth.
    Lower values indicate less risk.
    
    Args:
        prices: Series of prices
        period: Lookback period
    
    Returns:
        Series of Ulcer Index values
    """
    rolling_max = prices.rolling(window=period).max()
    percent_drawdown = 100 * (prices - rolling_max) / rolling_max
    squared_dd = percent_drawdown ** 2
    ulcer = np.sqrt(squared_dd.rolling(window=period).mean())
    return ulcer


# performance ratios

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                 periods_per_year: int = 252) -> float:
    """
    Sharpe Ratio.
    
    Risk-adjusted return measuring excess return per unit of total risk.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Trading periods per year (252 for daily, 365 for crypto)
    
    Returns:
        Annualized Sharpe ratio
    """
    returns_clean = returns.dropna()
    if len(returns_clean) == 0 or returns_clean.std() == 0:
        return np.nan
    
    # Convert annual risk-free to per-period
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    excess_returns = returns_clean - rf_per_period
    sharpe = excess_returns.mean() / excess_returns.std()
    
    # Annualize
    return sharpe * np.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                  target_return: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Sortino Ratio.
    
    Like Sharpe but only considers downside volatility.
    Better for asymmetric return distributions (crypto).
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        target_return: Minimum acceptable return (MAR)
        periods_per_year: Trading periods per year
    
    Returns:
        Annualized Sortino ratio
    """
    returns_clean = returns.dropna()
    if len(returns_clean) == 0:
        return np.nan
    
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    target_per_period = (1 + target_return) ** (1 / periods_per_year) - 1
    
    excess_returns = returns_clean - rf_per_period
    
    # Downside deviation (only negative returns below target)
    downside_returns = returns_clean[returns_clean < target_per_period] - target_per_period
    
    if len(downside_returns) == 0:
        return np.inf  # No downside risk
    
    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_std == 0:
        return np.nan
    
    sortino = excess_returns.mean() / downside_std
    return sortino * np.sqrt(periods_per_year)


def treynor_ratio(returns: pd.Series, benchmark_returns: pd.Series,
                  risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Treynor Ratio.
    
    Excess return per unit of systematic risk (beta).
    
    Args:
        returns: Series of portfolio returns
        benchmark_returns: Series of benchmark returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year
    
    Returns:
        Annualized Treynor ratio
    """
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Calculate beta
    beta_val = beta(returns, benchmark_returns)
    
    if beta_val == 0 or np.isnan(beta_val):
        return np.nan
    
    excess_return = (returns.mean() - rf_per_period) * periods_per_year
    return excess_return / beta_val


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series,
                      periods_per_year: int = 252) -> float:
    """
    Information Ratio.
    
    Active return relative to tracking error.
    
    Args:
        returns: Series of portfolio returns
        benchmark_returns: Series of benchmark returns
        periods_per_year: Trading periods per year
    
    Returns:
        Annualized Information ratio
    """
    active_returns = returns - benchmark_returns
    active_returns_clean = active_returns.dropna()
    
    if len(active_returns_clean) == 0 or active_returns_clean.std() == 0:
        return np.nan
    
    ir = active_returns_clean.mean() / active_returns_clean.std()
    return ir * np.sqrt(periods_per_year)


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Omega Ratio.
    
    Probability-weighted ratio of gains to losses.
    
    Args:
        returns: Series of returns
        threshold: Minimum acceptable return (default 0)
    
    Returns:
        Omega ratio (>1 is favorable)
    """
    returns_clean = returns.dropna()
    
    gains = returns_clean[returns_clean > threshold] - threshold
    losses = threshold - returns_clean[returns_clean <= threshold]
    
    if losses.sum() == 0:
        return np.inf
    
    return gains.sum() / losses.sum()


def gain_to_pain_ratio(returns: pd.Series) -> float:
    """
    Gain-to-Pain Ratio.
    
    Sum of returns divided by sum of absolute negative returns.
    
    Args:
        returns: Series of returns
    
    Returns:
        Gain-to-Pain ratio
    """
    returns_clean = returns.dropna()
    
    total_return = returns_clean.sum()
    pain = abs(returns_clean[returns_clean < 0].sum())
    
    if pain == 0:
        return np.inf
    
    return total_return / pain


def tail_ratio(returns: pd.Series, percentile: float = 5.0) -> float:
    """
    Tail Ratio.
    
    Ratio of right tail (gains) to left tail (losses).
    
    Args:
        returns: Series of returns
        percentile: Percentile for tail measurement (default 5%)
    
    Returns:
        Tail ratio (>1 indicates larger gains than losses in tails)
    """
    returns_clean = returns.dropna()
    
    right_tail = np.percentile(returns_clean, 100 - percentile)
    left_tail = np.percentile(returns_clean, percentile)
    
    if left_tail == 0:
        return np.nan
    
    return abs(right_tail / left_tail)


# systematic risk metrics

def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Beta (Market Risk).
    
    Measures systematic risk relative to a benchmark.
    
    Args:
        returns: Series of portfolio returns
        benchmark_returns: Series of benchmark returns
    
    Returns:
        Beta coefficient
    """
    # Align the series
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return np.nan
    
    port_returns = aligned.iloc[:, 0]
    bench_returns = aligned.iloc[:, 1]
    
    covariance = port_returns.cov(bench_returns)
    variance = bench_returns.var()
    
    if variance == 0:
        return np.nan
    
    return covariance / variance


def alpha(returns: pd.Series, benchmark_returns: pd.Series,
          risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Jensen's Alpha.
    
    Excess return not explained by beta (systematic risk).
    
    Args:
        returns: Series of portfolio returns
        benchmark_returns: Series of benchmark returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year
    
    Returns:
        Annualized alpha
    """
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    beta_val = beta(returns, benchmark_returns)
    if np.isnan(beta_val):
        return np.nan
    
    portfolio_return = returns.mean()
    benchmark_return = benchmark_returns.mean()
    
    alpha_per_period = portfolio_return - (rf_per_period + beta_val * (benchmark_return - rf_per_period))
    
    return alpha_per_period * periods_per_year


def r_squared(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    R-Squared (Coefficient of Determination).
    
    Proportion of variance explained by the benchmark.
    
    Args:
        returns: Series of portfolio returns
        benchmark_returns: Series of benchmark returns
    
    Returns:
        R-squared (0 to 1)
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return np.nan
    
    correlation = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    return correlation ** 2


def tracking_error(returns: pd.Series, benchmark_returns: pd.Series,
                   periods_per_year: int = 252) -> float:
    """
    Tracking Error.
    
    Standard deviation of active returns (portfolio - benchmark).
    
    Args:
        returns: Series of portfolio returns
        benchmark_returns: Series of benchmark returns
        periods_per_year: Trading periods per year
    
    Returns:
        Annualized tracking error
    """
    active_returns = returns - benchmark_returns
    te = active_returns.dropna().std() * np.sqrt(periods_per_year)
    return te


# more risk metrics

def volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualized Volatility.
    
    Args:
        returns: Series of returns
        periods_per_year: Trading periods per year
    
    Returns:
        Annualized volatility (standard deviation)
    """
    return returns.std() * np.sqrt(periods_per_year)


def downside_deviation(returns: pd.Series, target: float = 0.0,
                       periods_per_year: int = 252) -> float:
    """
    Downside Deviation.
    
    Standard deviation of returns below a target.
    
    Args:
        returns: Series of returns
        target: Target return (default 0)
        periods_per_year: Trading periods per year
    
    Returns:
        Annualized downside deviation
    """
    returns_clean = returns.dropna()
    downside = returns_clean[returns_clean < target] - target
    
    if len(downside) == 0:
        return 0.0
    
    dd = np.sqrt(np.mean(downside ** 2))
    return dd * np.sqrt(periods_per_year)


def skewness(returns: pd.Series) -> float:
    """
    Return Skewness.
    
    Measures asymmetry of return distribution.
    Negative skew indicates larger left tail (more extreme losses).
    
    Args:
        returns: Series of returns
    
    Returns:
        Skewness value
    """
    return returns.dropna().skew()


def kurtosis(returns: pd.Series) -> float:
    """
    Return Kurtosis (Excess).
    
    Measures tail fatness of distribution.
    Positive kurtosis indicates fatter tails (more extreme events).
    
    Args:
        returns: Series of returns
    
    Returns:
        Excess kurtosis value
    """
    return returns.dropna().kurtosis()


def win_rate(returns: pd.Series) -> float:
    """
    Win Rate.
    
    Percentage of positive returns.
    
    Args:
        returns: Series of returns
    
    Returns:
        Win rate as a decimal (0 to 1)
    """
    returns_clean = returns.dropna()
    if len(returns_clean) == 0:
        return np.nan
    return (returns_clean > 0).sum() / len(returns_clean)


def profit_factor(returns: pd.Series) -> float:
    """
    Profit Factor.
    
    Sum of profits divided by sum of losses.
    
    Args:
        returns: Series of returns
    
    Returns:
        Profit factor (>1 is profitable)
    """
    returns_clean = returns.dropna()
    
    profits = returns_clean[returns_clean > 0].sum()
    losses = abs(returns_clean[returns_clean < 0].sum())
    
    if losses == 0:
        return np.inf
    
    return profits / losses


def average_win_loss_ratio(returns: pd.Series) -> float:
    """
    Average Win/Loss Ratio.
    
    Average winning trade divided by average losing trade.
    
    Args:
        returns: Series of returns
    
    Returns:
        Win/Loss ratio
    """
    returns_clean = returns.dropna()
    
    avg_win = returns_clean[returns_clean > 0].mean()
    avg_loss = abs(returns_clean[returns_clean < 0].mean())
    
    if np.isnan(avg_loss) or avg_loss == 0:
        return np.nan
    
    return avg_win / avg_loss


def kelly_criterion(returns: pd.Series) -> float:
    """
    Kelly Criterion.
    
    Optimal fraction of capital to bet/invest.
    
    Args:
        returns: Series of returns
    
    Returns:
        Kelly fraction (can be >1, which suggests leverage)
    """
    returns_clean = returns.dropna()
    
    win_prob = win_rate(returns_clean)
    avg_win = returns_clean[returns_clean > 0].mean()
    avg_loss = abs(returns_clean[returns_clean < 0].mean())
    
    if np.isnan(avg_loss) or avg_loss == 0:
        return np.nan
    
    win_loss_ratio = avg_win / avg_loss
    
    # Kelly formula: f = (p * b - q) / b
    # where p = win probability, q = loss probability, b = win/loss ratio
    kelly = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
    
    return kelly


# risk metrics snapshot dataclass

@dataclass
class RiskMetricsSnapshot:
    """Comprehensive snapshot of all risk and performance metrics."""
    
    # Value at Risk
    var_95_hist: Optional[float] = None
    var_99_hist: Optional[float] = None
    var_95_param: Optional[float] = None
    var_95_cf: Optional[float] = None  # Cornish-Fisher adjusted
    
    # Expected Shortfall
    es_95: Optional[float] = None
    es_99: Optional[float] = None
    
    # Drawdown
    max_drawdown: Optional[float] = None
    current_drawdown: Optional[float] = None
    max_drawdown_duration: Optional[int] = None
    
    # Performance Ratios
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    omega_ratio: Optional[float] = None
    
    # Benchmark-relative (if applicable)
    treynor_ratio: Optional[float] = None
    information_ratio: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    r_squared: Optional[float] = None
    
    # Distribution Stats
    annualized_volatility: Optional[float] = None
    downside_deviation: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    # Trading Stats
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    avg_win_loss_ratio: Optional[float] = None
    kelly_criterion: Optional[float] = None
    
    # Summary
    risk_rating: Optional[str] = None  # "low", "medium", "high", "extreme"


class RiskAnalyzer:
    """
    Comprehensive risk analysis engine for stocks and crypto portfolios.
    
    Computes VaR, ES, drawdowns, and all performance ratios.
    """
    
    def __init__(self, prices: pd.Series, benchmark_prices: Optional[pd.Series] = None,
                 risk_free_rate: float = 0.0, periods_per_year: int = 252):
        """
        Initialize Risk Analyzer.
        
        Args:
            prices: Series of portfolio prices/values
            benchmark_prices: Optional benchmark prices for relative metrics
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year (252 for stocks, 365 for crypto)
        """
        self.prices = prices.dropna()
        self.returns = calculate_returns(prices)
        self.benchmark_prices = benchmark_prices
        self.benchmark_returns = calculate_returns(benchmark_prices) if benchmark_prices is not None else None
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
    
    def get_snapshot(self) -> RiskMetricsSnapshot:
        """
        Compute all risk metrics and return a snapshot.
        
        Returns:
            RiskMetricsSnapshot with all computed metrics
        """
        r = self.returns
        p = self.prices
        
        # VaR calculations
        var_95 = var_historical(r, 0.95)
        var_99 = var_historical(r, 0.99)
        
        try:
            var_95_p = var_parametric(r, 0.95)
        except ImportError:
            var_95_p = None
        
        try:
            var_95_cf = var_cornish_fisher(r, 0.95)
        except ImportError:
            var_95_cf = None
        
        # Expected Shortfall
        es_95 = expected_shortfall(r, 0.95)
        es_99 = expected_shortfall(r, 0.99)
        
        # Drawdown analysis
        mdd = max_drawdown(p)
        current_dd = drawdown(p).iloc[-1] if len(p) > 0 else None
        max_dd_dur, _ = drawdown_duration(p)
        
        # Performance ratios
        sr = sharpe_ratio(r, self.risk_free_rate, self.periods_per_year)
        sort = sortino_ratio(r, self.risk_free_rate, 0.0, self.periods_per_year)
        cal = calmar_ratio(r, p, self.periods_per_year)
        omg = omega_ratio(r)
        
        # Benchmark-relative metrics
        tr, ir, alph, bet, rsq = None, None, None, None, None
        if self.benchmark_returns is not None:
            tr = treynor_ratio(r, self.benchmark_returns, self.risk_free_rate, self.periods_per_year)
            ir = information_ratio(r, self.benchmark_returns, self.periods_per_year)
            alph = alpha(r, self.benchmark_returns, self.risk_free_rate, self.periods_per_year)
            bet = beta(r, self.benchmark_returns)
            rsq = r_squared(r, self.benchmark_returns)
        
        # Distribution stats
        vol = volatility(r, self.periods_per_year)
        dd_dev = downside_deviation(r, 0.0, self.periods_per_year)
        skew = skewness(r)
        kurt = kurtosis(r)
        
        # Trading stats
        wr = win_rate(r)
        pf = profit_factor(r)
        awl = average_win_loss_ratio(r)
        kelly = kelly_criterion(r)
        
        # Risk rating based on multiple factors
        risk_rating = self._compute_risk_rating(var_95, mdd, vol, skew, kurt)
        
        return RiskMetricsSnapshot(
            var_95_hist=var_95,
            var_99_hist=var_99,
            var_95_param=var_95_p,
            var_95_cf=var_95_cf,
            es_95=es_95,
            es_99=es_99,
            max_drawdown=mdd,
            current_drawdown=abs(current_dd) if current_dd else None,
            max_drawdown_duration=int(max_dd_dur) if max_dd_dur else None,
            sharpe_ratio=sr,
            sortino_ratio=sort,
            calmar_ratio=cal,
            omega_ratio=omg,
            treynor_ratio=tr,
            information_ratio=ir,
            alpha=alph,
            beta=bet,
            r_squared=rsq,
            annualized_volatility=vol,
            downside_deviation=dd_dev,
            skewness=skew,
            kurtosis=kurt,
            win_rate=wr,
            profit_factor=pf,
            avg_win_loss_ratio=awl,
            kelly_criterion=kelly,
            risk_rating=risk_rating
        )
    
    def _compute_risk_rating(self, var_95: float, mdd: float, vol: float,
                             skew: float, kurt: float) -> str:
        """
        Compute overall risk rating based on multiple factors.
        
        Returns:
            Risk rating: "low", "medium", "high", "extreme"
        """
        score = 0
        
        # VaR contribution (daily)
        if var_95 is not None and not np.isnan(var_95):
            if var_95 > 0.05:  # >5% daily VaR
                score += 3
            elif var_95 > 0.03:
                score += 2
            elif var_95 > 0.02:
                score += 1
        
        # Max Drawdown contribution
        if mdd is not None and not np.isnan(mdd):
            if mdd > 0.50:  # >50% drawdown
                score += 3
            elif mdd > 0.30:
                score += 2
            elif mdd > 0.15:
                score += 1
        
        # Volatility contribution (annualized)
        if vol is not None and not np.isnan(vol):
            if vol > 0.80:  # >80% annual vol
                score += 3
            elif vol > 0.50:
                score += 2
            elif vol > 0.30:
                score += 1
        
        # Negative skew contribution
        if skew is not None and not np.isnan(skew):
            if skew < -1:
                score += 2
            elif skew < -0.5:
                score += 1
        
        # Fat tails contribution
        if kurt is not None and not np.isnan(kurt):
            if kurt > 6:
                score += 2
            elif kurt > 3:
                score += 1
        
        # Rating based on total score
        if score >= 10:
            return "extreme"
        elif score >= 6:
            return "high"
        elif score >= 3:
            return "medium"
        else:
            return "low"
    
    def generate_report(self) -> str:
        """
        Generate a text report of risk metrics.
        
        Returns:
            Formatted string report
        """
        snapshot = self.get_snapshot()
        
        report = """
╔══════════════════════════════════════════════════════════════════╗
║                    RISK ANALYSIS REPORT                         ║
╠══════════════════════════════════════════════════════════════════╣

📊 VALUE AT RISK (VaR)
├── 95% Historical VaR:     {var_95:.2%}
├── 99% Historical VaR:     {var_99:.2%}
├── 95% Parametric VaR:     {var_95_p}
└── 95% Cornish-Fisher VaR: {var_95_cf}

📉 EXPECTED SHORTFALL (CVaR)
├── 95% ES: {es_95:.2%}
└── 99% ES: {es_99:.2%}

📈 DRAWDOWN ANALYSIS
├── Maximum Drawdown:     {mdd:.2%}
├── Current Drawdown:     {curr_dd}
└── Max DD Duration:      {dd_dur} periods

⭐ PERFORMANCE RATIOS
├── Sharpe Ratio:    {sharpe:.3f}
├── Sortino Ratio:   {sortino:.3f}
├── Calmar Ratio:    {calmar:.3f}
└── Omega Ratio:     {omega:.3f}

📊 DISTRIBUTION STATISTICS
├── Annualized Vol:      {vol:.2%}
├── Downside Deviation:  {dd_dev:.2%}
├── Skewness:            {skew:.3f}
└── Kurtosis:            {kurt:.3f}

🎯 TRADING STATISTICS
├── Win Rate:           {wr:.2%}
├── Profit Factor:      {pf:.3f}
├── Avg Win/Loss:       {awl:.3f}
└── Kelly Criterion:    {kelly:.3f}

⚠️  RISK RATING: {rating}

╚══════════════════════════════════════════════════════════════════╝
""".format(
            var_95=snapshot.var_95_hist or 0,
            var_99=snapshot.var_99_hist or 0,
            var_95_p=f"{snapshot.var_95_param:.2%}" if snapshot.var_95_param else "N/A",
            var_95_cf=f"{snapshot.var_95_cf:.2%}" if snapshot.var_95_cf else "N/A",
            es_95=snapshot.es_95 or 0,
            es_99=snapshot.es_99 or 0,
            mdd=snapshot.max_drawdown or 0,
            curr_dd=f"{snapshot.current_drawdown:.2%}" if snapshot.current_drawdown else "N/A",
            dd_dur=snapshot.max_drawdown_duration or "N/A",
            sharpe=snapshot.sharpe_ratio or 0,
            sortino=snapshot.sortino_ratio or 0,
            calmar=snapshot.calmar_ratio or 0,
            omega=snapshot.omega_ratio or 0,
            vol=snapshot.annualized_volatility or 0,
            dd_dev=snapshot.downside_deviation or 0,
            skew=snapshot.skewness or 0,
            kurt=snapshot.kurtosis or 0,
            wr=snapshot.win_rate or 0,
            pf=snapshot.profit_factor or 0,
            awl=snapshot.avg_win_loss_ratio or 0,
            kelly=snapshot.kelly_criterion or 0,
            rating=snapshot.risk_rating or "N/A"
        )
        
        return report


# position sizing & risk budgeting

def position_size_fixed_risk(account_value: float, risk_per_trade: float,
                             entry_price: float, stop_loss_price: float) -> float:
    """
    Calculate position size based on fixed risk per trade.
    
    Args:
        account_value: Total account value
        risk_per_trade: Maximum risk per trade (e.g., 0.02 for 2%)
        entry_price: Entry price
        stop_loss_price: Stop loss price
    
    Returns:
        Number of shares/units to trade
    """
    risk_amount = account_value * risk_per_trade
    risk_per_share = abs(entry_price - stop_loss_price)
    
    if risk_per_share == 0:
        return 0
    
    return risk_amount / risk_per_share


def position_size_volatility(account_value: float, target_risk: float,
                             atr_value: float, atr_multiplier: float = 2.0) -> float:
    """
    Position size based on volatility (ATR).
    
    Args:
        account_value: Total account value
        target_risk: Target risk as decimal (e.g., 0.02 for 2%)
        atr_value: Current ATR value
        atr_multiplier: Multiplier for stop distance (default 2x ATR)
    
    Returns:
        Number of shares/units to trade
    """
    risk_amount = account_value * target_risk
    risk_per_share = atr_value * atr_multiplier
    
    if risk_per_share == 0:
        return 0
    
    return risk_amount / risk_per_share


def position_size_kelly(returns: pd.Series, account_value: float,
                        fraction: float = 0.5) -> float:
    """
    Position size based on Kelly Criterion.
    
    Args:
        returns: Historical returns for Kelly calculation
        account_value: Total account value
        fraction: Kelly fraction (0.5 = half Kelly for safety)
    
    Returns:
        Position size as dollar amount
    """
    kelly = kelly_criterion(returns)
    
    if np.isnan(kelly) or kelly <= 0:
        return 0
    
    # Apply fraction (half Kelly is common for safety)
    adjusted_kelly = kelly * fraction
    
    # Cap at 100% of account (no leverage by default)
    adjusted_kelly = min(adjusted_kelly, 1.0)
    
    return account_value * adjusted_kelly


def optimal_f(returns: pd.Series) -> float:
    """
    Optimal f (Ralph Vince).
    
    Finds the optimal fraction of capital to risk to maximize growth.
    
    Args:
        returns: Series of trade returns (not portfolio returns)
    
    Returns:
        Optimal fraction to risk
    """
    returns_clean = returns.dropna()
    if len(returns_clean) == 0:
        return 0
    
    max_loss = abs(returns_clean.min())
    if max_loss == 0:
        return 0
    
    # Binary search for optimal f
    best_f = 0
    best_twwr = 0
    
    for f in np.arange(0.01, 1.01, 0.01):
        hpr = 1 + f * (-returns_clean / max_loss)
        if (hpr <= 0).any():
            continue
        twwr = np.prod(hpr) ** (1 / len(hpr))
        if twwr > best_twwr:
            best_twwr = twwr
            best_f = f
    
    return best_f
