import asyncio
import logging
from .live_bot import run

def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logging.info("Shutting down (Ctrl+C). Bye!")

if __name__ == "__main__":
    main()
