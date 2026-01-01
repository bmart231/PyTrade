from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


@dataclass(frozen=True)
class Bar:
    # one minute bar
    symbol: str # ticker symbol (string)
    ts: datetime # timestamp
    open: float # open price
    high: float # high price
    low: float # low price
    close: float # close price
    volume: float # volume

# takes the ticker symbol and side (buy/sell) of the order
class Side(str, Enum):
    BUY = "BUY" # buy side
    SELL = "SELL" # sell side

# Takes the order type (market/limit)
class OrderType(str, Enum):
    MARKET = "MARKET" # market order
    LIMIT = "LIMIT" # limit order

# Data class for order types and what to trade
@dataclass(frozen=True)
class Order:
    symbol: str # ticker symbol (string)
    ts: datetime # timestamp
    side: Side # buy or sell
    qty: int # quantity
    order_type: OrderType = OrderType.MARKET # market or limit
    limit_price: Optional[float] = None  # used only for LIMIT
    
    # validation after initialization to ensure no negative value
    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError("Order.qty must be positive")

        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit_price required for LIMIT orders")

        if self.order_type == OrderType.MARKET and self.limit_price is not None:
            raise ValueError("limit_price should be None for MARKET orders")

# Data class for trade fills
@dataclass(frozen=True)
class Fill:
    symbol: str # ticker symbol (string)
    ts: datetime # timestamp
    side: Side # buy or sell
    qty: int # quantity
    price: float # fill price
    fee: float = 0.0 # transaction fee

# Data class for position and portfolio
@dataclass
class Position:
    symbol: str # ticker symbol (string)
    qty: int = 0 # quantity
    avg_price: float = 0.0  # cost basis

# portfolio with cash and position
@dataclass
class Portfolio:
    cash: float # available cash
    position: Position # current position

    # calculate current equity
    def equity(self, mark_price: float) -> float:
        """Cash + marked-to-market value of the position."""
        return self.cash + self.position.qty * mark_price