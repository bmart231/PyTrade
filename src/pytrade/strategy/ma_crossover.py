from __future__ import annotations
from .base import Decision, Signal, Strategy

# Makes the moving average crossover strategy that extends the base Strategy
class MA_Crossover_Strategy(Strategy):
    # initialize with threshold for crossover detection
    def __init__(self, threshold: float = 0.01):
        self.threshold = float(threshold)

    # implement decide method to generate trading signals by using moving averages (fast, slow)
    def decide(self, features) -> Decision:
        sma_fast = features.sma_fast
        sma_slow = features.sma_slow

        # if there is nothing in the sma values
        if sma_fast is None or sma_slow is None:
            return Decision(signal=Signal.HOLD, reason="Indicators not ready")
        # avoid division by zero
        if sma_slow == 0:
            return Decision(signal=Signal.HOLD, reason="Invalid sma_slow=0")

        # get the difference between the two moving averages
        diff = sma_fast - sma_slow
        # get the relative difference
        rel_diff = diff / sma_slow

        # generate signals based on the relative difference and threshold
        if rel_diff > self.threshold:
            return Decision(signal=Signal.BUY, reason="Fast MA above Slow MA")
        if rel_diff < -self.threshold:
            return Decision(signal=Signal.SELL, reason="Fast MA below Slow MA")
        # if there is no significant crossover then just hold
        return Decision(signal=Signal.HOLD, reason="No significant crossover")
