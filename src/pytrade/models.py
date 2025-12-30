# data structures for PyTrade application
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

@dataclass(frozen=True)
class Bar:
    # one minute bar
    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass(frozen=True)
class Order:
    symbol: str
    ts: datetime
    side: Side
    qty: int
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None  # used only for LIMIT


@dataclass(frozen=True)
class Fill:
    symbol: str
    ts: datetime
    side: Side
    qty: int
    price: float
    fee: float = 0.0


@dataclass
class Position:
    symbol: str
    qty: int = 0
    avg_price: float = 0.0  # cost basis


@dataclass
class Portfolio:
    cash: float
    position: Position

    def equity(self, mark_price: float) -> float:
        """Cash + marked-to-market value of the position."""
        return self.cash + self.position.qty * mark_price