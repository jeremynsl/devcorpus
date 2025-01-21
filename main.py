import asyncio
from logger import logger
import os
import sys
import json
from config import load_config, CONFIG_FILE
import argparse
from scraper import scrape_recursive

###############################################################################
# Pause/Resume Support
###############################################################################
is_paused_event = asyncio.Event()
is_paused_event.set()  # 'set' means "not paused"

async def watch_for_input():
    """
    In a separate task, listens for user input and toggles pause/resume
    or quits (q) if requested.
    """
    loop = asyncio.get_event_loop()
    logger.info("Type 'p' to pause, 'r' to resume, or 'q' to quit.")
    while True:
        # Read a line in a thread to avoid blocking the event loop
        user_input = await loop.run_in_executor(None, sys.stdin.readline)
        user_input = user_input.strip().lower()
        if user_input == "p":
            logger.info("Pausing scraping.")
            is_paused_event.clear()
        elif user_input == "r":
            logger.info("Resuming scraping.")
            is_paused_event.set()
        elif user_input == "q":
            logger.info("Stopping scraping.")
            # Force an exit from the entire program
            os._exit(0)


###############################################################################
# Main Entry
###############################################################################
def main():
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(description='Recursive web scraper and chat interface')
    
    # URL can be either a full URL for scraping or just the DB name for chat
    parser.add_argument('target', nargs='?', help='URL to scrape or database name for chat')
    parser.add_argument('-db', '--database', action='store_true', 
                      help='Store scraped content in ChromaDB')
    parser.add_argument('--chat', action='store_true',
                      help='Start chat interface')
    parser.add_argument('--model',
                      default='gemini/gemini-1.5-flash',
                      help='LiteLLM model string (e.g., gemini/gemini-1.5-flash, gpt-3.5-turbo, claude-3-sonnet)')
    args = parser.parse_args()

    # If no arguments provided, launch Gradio interface
    if not args.target and not args.chat:
        from gradio_app import create_demo
        demo = create_demo()
        demo.launch()
        return

    # Determine mode based on arguments
    is_url = args.target and args.target.startswith(('http://', 'https://'))
    
    if is_url:
        # Scraping mode
        if args.chat and not args.database:
            logger.error("Chat with scraping requires --database flag")
            sys.exit(1)
            
        # Load config for scraping
        proxies, rate_limit, user_agent = load_config(CONFIG_FILE)
        global proxies_list
        proxies_list = proxies

        # Start the async environment
        async def run():
            # Start a background task to watch for keyboard input
            input_task = asyncio.create_task(watch_for_input())

            # Run the scraper
            await scrape_recursive(args.target, user_agent, rate_limit, args.database)

            # Start chat if requested
            if args.chat:
                from gradio_app import create_demo
                demo = create_demo()
                demo.launch()

            # Cancel the input watcher
            input_task.cancel()
            try:
                await input_task
            except asyncio.CancelledError:
                pass
            os._exit(0)

        asyncio.run(run())
    else:
        # Chat-only mode
        if args.database:
            logger.warning("--database flag ignored in chat-only mode")
            
        if args.chat:
            # Launch terminal chat interface
            from chat import ChatInterface
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
            db_name = f"{args.target}.txt"  # Add .txt as chat expects the text filename
            chat = ChatInterface(
                db_name,
                model=config["chat"]["models"]["default"]
            )
            chat.run_chat_loop()
        else:
            # Launch Gradio interface with pre-selected database
            from gradio_app import create_demo
            demo = create_demo()
            demo.launch()

if __name__ == "__main__":
    main()
