import asyncio
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

async def run():
    logging.info("live_bot starting...")
    i = 0
    while True:
        i += 1
        logging.info("heartbeat %d", i)
        await asyncio.sleep(5)
