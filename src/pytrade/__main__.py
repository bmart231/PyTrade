import asyncio
import logging

from .live_bot import run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logging.info("Shutting down (Ctrl+C). Bye!")

if __name__ == "__main__":
    main()
