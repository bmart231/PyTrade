from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import numpy as np # import Bar class (Bar.py) to use as input
from .bars import Bar

# Goal: compute rolling indicators from incoming bars created in Data.py

@dataclass(frozen=True)
class FeatureSnapshot:
    """Snapshot of computed features at a point in time."""
    symbol: str # ticker symbol (string)
    ts: object  # datetime, but keep flexible if you want
    close: float # close price

    # indicators
    sma_fast: float | None # simple moving average (fast)
    sma_slow: float | None # simple moving average (slow)
    ret_1: float | None # last return
    vol: float | None # rolling volatility

    ready: bool # are indicators ready?

class FeatureEngine:
    """Compute rolling features from incoming bars."""
    # initialize with parameters for indicators
    
    def __init__(self, fast: int = 10, slow: int = 30, vol_window: int = 30):
        if fast <= 1 or slow <= 1: # speed for computing
            raise ValueError("fast and slow must be > 1")
        if fast >= slow: # speed for computing
            raise ValueError("fast must be < slow")

        self.fast = fast # get fast value
        self.slow = slow # get slow value
        self.vol_window = vol_window # get volatility window

        self.rolling_closes = deque(maxlen=slow) # store close prices
        self.rolling_returns = deque(maxlen=vol_window) # store returns

        self.last_bar: Bar | None = None # last bar seen
        self.last_close: float | None = None # last close price

    def update(self, bar: Bar) -> None:
        """Push a new bar into the rolling windows."""
        current_close_price = float(bar.close)
        previous_close_price = self.last_close

        # compute 1-step return
        if previous_close_price is not None: 
            # return calculation
            returns = (current_close_price / previous_close_price) - 1.0
            # append return to deque 
            self.rolling_returns.append(returns)

        # append close price to deque
        self.rolling_closes.append(current_close_price) # appen close price
        self.last_close = current_close_price # update last close price
        self.last_bar = bar # update last bar seen

    # computes the current features from rolling windows 
    def snapshot(self) -> FeatureSnapshot:
        """Compute current features from rolling windows."""
        if self.last_bar is None:
            return FeatureSnapshot(
                # setting everything to unknown / None
                symbol="", # unknown symbol 
                ts=None, # unknown timestamp
                close=float("nan"), # unknown close price
                sma_fast=None,
                sma_slow=None,
                ret_1=None,
                vol=None,
                ready=False,
            )

        # SMAs
        close_price_indexes = list(self.rolling_closes)
        length = len(close_price_indexes) # length of close prices indexes
        

        sma_fast = np.float64(np.mean(close_price_indexes[-self.fast:])) if length >= self.fast else None
        sma_slow = np.float64(np.mean(close_price_indexes[-self.slow:])) if length >= self.slow else None

        # last return
        last_return = self.rolling_returns[-1] if len(self.rolling_returns) >= 1 else None
        # rolling volatility of returns (scaled by sqrt(n) as a rough magnitude)
        vol = None
        # ensure enough data points for volatility
        if len(self.rolling_returns) >= max(5, self.vol_window // 3):
            # volatility calculation
            vol = np.std(self.rolling_returns) * np.sqrt(len(self.rolling_returns))
        # are indicators ready?
        ready = (sma_fast is not None) and (sma_slow is not None)

        # returns the current snapshot of features with all the computed values
        return FeatureSnapshot(
            symbol=self.last_bar.symbol,
            ts=self.last_bar.ts,
            close=float(self.last_bar.close),
            sma_fast=sma_fast,
            sma_slow=sma_slow,
            ret_1=last_return,
            vol=np.float64(vol) if vol is not None else None,
            ready=ready,
        )
