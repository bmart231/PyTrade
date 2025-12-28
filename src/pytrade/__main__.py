import asyncio
from .live_bot import run  
def main():
    result = run()
    if asyncio.iscoroutine(result):
        asyncio.run(result)

if __name__ == "__main__":
    main()
