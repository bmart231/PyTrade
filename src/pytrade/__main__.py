import asyncio
from .live_bot import run

# Main function to start the asyncio event loop
def main() -> None:
    asyncio.run(run())

# Entry point for the application
if __name__ == "__main__":
    main()
