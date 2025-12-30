from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import sqrt
from statistics import mean, pstdev

# import Bar class (Bar.py) to use as input
from .Bars import Bar

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

        self._closes = deque(maxlen=slow) # store close prices
        self._rets = deque(maxlen=vol_window) # store returns

        self._last_bar: Bar | None = None # last bar seen
        self._last_close: float | None = None # last close price

    def update(self, bar: Bar) -> None:
        """Push a new bar into the rolling windows."""
        c = float(bar.close)

        # compute 1-step return
        if self._last_close is not None:
            r = (c / self._last_close) - 1.0
            self._rets.append(r)

        self._closes.append(c)
        self._last_close = c
        self._last_bar = bar

    def snapshot(self) -> FeatureSnapshot:
        """Compute current features from rolling windows."""
        if self._last_bar is None:
            return FeatureSnapshot(
                symbol="",
                ts=None,
                close=float("nan"),
                sma_fast=None,
                sma_slow=None,
                ret_1=None,
                vol=None,
                ready=False,
            )

        # SMAs
        sma_fast = mean(list(self._closes)[-self.fast:]) if len(self._closes) >= self.fast else None
        sma_slow = mean(self._closes) if len(self._closes) >= self.slow else None

        # last return
        ret_1 = self._rets[-1] if len(self._rets) >= 1 else None

        # rolling volatility of returns (scaled by sqrt(n) as a rough magnitude)
        vol = None
        if len(self._rets) >= max(5, self.vol_window // 3):
            vol = pstdev(self._rets) * sqrt(len(self._rets))

        ready = (sma_fast is not None) and (sma_slow is not None)

        return FeatureSnapshot(
            symbol=self._last_bar.symbol,
            ts=self._last_bar.ts,
            close=float(self._last_bar.close),
            sma_fast=sma_fast,
            sma_slow=sma_slow,
            ret_1=ret_1,
            vol=vol,
            ready=ready,
        )
