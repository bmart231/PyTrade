from __future__ import annotations

import numpy as np
import asyncio
import logging
from datetime import datetime, timezone

from .features import FeatureEngine
from .models import Bar
from .strategy.ma_crossover import MA_Crossover_Strategy

# simple live trading bot that uses FeatureEngine and MA_Crossover_Strategy
async def run() -> None:
    logging.info("Beginning Live Bot...") # log startup messsage

    features = FeatureEngine(fast=10, slow=30, vol_window=30)  # initialize FeatureEngine
    strategy = MA_Crossover_Strategy(threshold=0.01)           # initialize strategy

    close = 100.0  # starting mock price

    while True:
        # simple mock price movement
        close += 0.1

        bar = Bar(
            symbol="TEST",
            ts=datetime.now(timezone.utc),  # real datetime timestamp
            open=close - 0.2, # open price
            high=close + 0.3, # high price
            low=close - 0.4, # low price
            close=close, # close price
            volume=1000.0, # volume
        )

        features.update(bar)             # update features with new bar
        snapshot = features.snapshot()   # get feature snapshot
        decision = strategy.decide(snapshot)  # get trading decision

        # log the bar info, features, and decision made
        logging.info(
            bar.ts, # timestamp
            bar.close, # close price
            None if snapshot.sma_fast is None else f"{np.round(snapshot.sma_fast, 4)}", # SMA fast
            None if snapshot.sma_slow is None else f"{np.round(snapshot.sma_slow, 4)}", # SMA slow
            decision.signal, # trading signal
            decision.reason, # reason for decision
        )

        await asyncio.sleep(5)  # sleep for 5 seconds
