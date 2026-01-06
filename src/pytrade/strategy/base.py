# defines the “contract” for any strategy. A strategy must implement the on_bar method to process incoming bars.
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# defines the possible trading signals
class Signal(str, Enum):
    BUY = "BUY" # buy signal 
    SELL = "SELL" # sell signal
    HOLD = "HOLD" # hold signal 


# decision output from a strategy, takes in signal and reason
@dataclass(frozen=True)
class Decision:
    signal: Signal # trading signal (Buy, Sell, Hold)
    reason: str # reason (string)


# base strategy interface (all strategies should implement decide)
class Strategy:
    def decide(self, features) -> Decision:
        """
        Given a FeatureSnapshot (from features.py), return a Decision.
        This base method is meant to be overridden by concrete strategies.
        """
        raise NotImplementedError("Strategy.decide must be implemented by subclasses")
