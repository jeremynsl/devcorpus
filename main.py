import asyncio
import logging
import os
import sys
import json
from urllib.parse import urljoin, urlparse, urlunparse
import httpx
from bs4 import BeautifulSoup
from trafilatura import extract
from trafilatura.settings import use_config
from trafilatura.core import extract_metadata
from tqdm import tqdm
from chroma import ChromaHandler
import argparse
from text_processor import TextProcessor

# Initialize text processor as a singleton
text_processor = TextProcessor()

###############################################################################
# Global Config / Constants
###############################################################################
CONFIG_FILE = "scraper_config.json"
LOG_FILE_NAME = "scraper.log"

def get_output_filename(url: str) -> str:
    """
    Convert URL to a valid filename by removing special characters and slashes.
    Example: https://svelte.dev/docs/ becomes httpssveltedevdocs.txt
    """
    parsed = urlparse(url)
    # Create scrapedtxt directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), "scrapedtxt")
    os.makedirs(output_dir, exist_ok=True)
    
    # Use domain name as filename, removing special characters
    filename = parsed.netloc.replace(".", "_").replace("-", "_")
    if not filename:
        filename = "default"
    return os.path.join(output_dir, f"{filename}.txt")

###############################################################################
# Logging Setup
###############################################################################
logger = logging.getLogger("RecursiveScraper")
logger.setLevel(logging.DEBUG)

log_formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)

# File handler
file_handler = logging.FileHandler(LOG_FILE_NAME, mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(log_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

#httpx_logger = logging.getLogger("httpx")
#httpx_logger.setLevel(logging.DEBUG)
#httpx_logger.addHandler(console_handler)
#httpx_logger.addHandler(file_handler)

###############################################################################
# Load Config
###############################################################################
def load_config(config_path: str):
    """
    Load the JSON config file, which should contain:
        {
          "proxies": ["http://proxy1:8080", "http://proxy2:8080"],
          "rate_limit": 5,
          "user_agent": "MyScraperBot/1.0"
        }
    """
    if not os.path.exists(config_path):
        logger.error(f"Config file {config_path} not found.")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    proxies = data.get("proxies", [])
    rate_limit = data.get("rate_limit", 5)
    user_agent = data.get("user_agent", "MyScraperBot/1.0")

    return proxies, rate_limit, user_agent

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
# Proxy Failover
###############################################################################
proxy_lock = asyncio.Lock()
current_proxy_index = 0
proxies_list = []  # Will be populated after loading config

def remove_anchor(url: str) -> str:
    """
    Remove the anchor/fragment from a URL so that
    http://example.com/page#something == http://example.com/page
    """
    parsed = urlparse(url)
    # Replace the fragment with an empty string
    parsed = parsed._replace(fragment="")
    return urlunparse(parsed)

def get_domain(url: str) -> str:
    return urlparse(url).netloc

async def get_next_proxy():
    """
    Returns the currently selected proxy from the proxies_list.
    If none exist, return None.
    """
    async with proxy_lock:
        if not proxies_list:
            return None
        return proxies_list[current_proxy_index]

async def switch_to_next_proxy():
    """
    Rotate to the next proxy in proxies_list (for failover).
    """
    async with proxy_lock:
        global current_proxy_index
        if not proxies_list:
            return
        current_proxy_index = (current_proxy_index + 1) % len(proxies_list)
        logger.warning(f"Switching to next proxy: {proxies_list[current_proxy_index]}")

###############################################################################
# Fetching / Parsing
###############################################################################
async def fetch_page(url: str, user_agent: str) -> str:
    """
    Attempt to fetch the given URL (HTML) with up to len(proxies_list) retries
    if proxies are available and we encounter certain HTTP errors (e.g., 403).
    """
    max_retries = max(1, len(proxies_list) or 1)  # at least 1 attempt
    attempts = 0

    while attempts < max_retries:
        attempts += 1
        proxy_url = await get_next_proxy()
        logger.debug(f"Fetching {url} [Attempt {attempts}] (Proxy={proxy_url})")

        # Prepare client initialization arguments
        client_args = {
            "headers": {"User-Agent": user_agent},
            "timeout": 30.0,
            "follow_redirects": True,
        }
        # If a proxy URL is available, add it to the arguments
        if proxy_url:
            client_args["proxy"] = proxy_url

        async with httpx.AsyncClient(**client_args) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.text
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                # If a "block" or "forbidden", try switching proxy
                if status_code in (403, 429):
                    logger.warning(
                        f"Received status {status_code} for {url}. "
                        "Attempting to switch proxy and retry."
                    )
                    await switch_to_next_proxy()
                else:
                    logger.error(f"HTTP error {status_code} for {url}: {e}")
                    return ""
            except httpx.HTTPError as e:
                # Could be ConnectionError, Timeout, etc.
                logger.error(f"Failed to fetch {url}: {e}")
                await switch_to_next_proxy()

    logger.error(f"All proxies failed or request error for {url}. Giving up.")
    return ""

def extract_links(html: str, base_url: str, domain: str, start_path: str) -> list[str]:
    """
    Extract all in-domain links from the HTML using trafilatura.
    Falls back to BeautifulSoup if trafilatura doesn't find links.
    Note that we remove anchors for duplicate-checking.
    """
    # Try trafilatura first
    metadata = extract_metadata(html, default_url=base_url)
    if metadata:
        links = metadata.as_dict().get('links', [])
        if links:
            filtered_links = []
            for link in links:
                link_no_anchor = remove_anchor(link)
                parsed_link = urlparse(link_no_anchor)
                if parsed_link.netloc == domain and parsed_link.path.startswith(start_path):
                    filtered_links.append(link_no_anchor)
            return filtered_links
    
    # Fallback to BeautifulSoup if trafilatura found no links
    soup = BeautifulSoup(html, "html.parser")
    found_links = []
    for a_tag in soup.find_all("a", href=True):
        raw_link = a_tag["href"]
        absolute_link = urljoin(base_url, raw_link)
        absolute_link_no_anchor = remove_anchor(absolute_link)
        
        # Parse link and check both domain and path
        parsed_link = urlparse(absolute_link_no_anchor)
        if parsed_link.netloc == domain and parsed_link.path.startswith(start_path):
            found_links.append(absolute_link_no_anchor)
            
    return found_links

def html_to_markdown(html: str) -> str:
    """
    Convert HTML to Markdown text using trafilatura.
    First remove boilerplate elements, then extract text.
    """
    # Remove common boilerplate elements
    cleaned_html = text_processor.preprocess_html(html)
    
    # Configure trafilatura
    config = use_config()
    config.set("DEFAULT", "include_links", "False")
    config.set("DEFAULT", "include_formatting", "True")
    config.set("DEFAULT", "output_format", "markdown")
    config.set("DEFAULT", "extraction_timeout", "0")
    
    # Try trafilatura first
    text = extract(cleaned_html, config=config)
    
    # Fall back to BeautifulSoup if needed
    if not text:
        logger.warning("Trafilatura extraction failed, falling back to BeautifulSoup")
        soup = BeautifulSoup(cleaned_html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
    
    return text or ""  # Ensure we never return None

async def scrape_recursive(start_url: str, user_agent: str, rate_limit: int, use_db: bool = False):
    """
    Recursively scrape all links from start_url domain.
    Writes results incrementally to a file in the scrapedtxt directory.
    If use_db is True, also stores content in ChromaDB.
    """
    output_file = get_output_filename(start_url)
    domain = get_domain(start_url)
    parsed = urlparse(start_url)
    start_path = parsed.path
    if not start_path.endswith("/"):
        start_path += "/"
        
    visited = set()
    to_visit = asyncio.Queue()

    # Initialize ChromaDB if requested
    db_handler = None
    if use_db:
        # Get collection name from URL
        collection_name = ChromaHandler.get_collection_name(start_url)
        db_handler = ChromaHandler(collection_name)

    # Put the initial URL in the queue
    await to_visit.put(remove_anchor(start_url))

    # Rate limiter (max concurrency)
    semaphore = asyncio.Semaphore(rate_limit)

    # Create a progress bar (indeterminate total)
    progress_bar = tqdm(desc="Pages Scraped", unit="pages", total=0)
    
    def update_progress():
        """Log the current progress"""
        logger.info(f"Pages Scraped: {progress_bar.n} pages [{progress_bar.format_dict['rate']:.2f} pages/s]")

    # We'll keep the file open in "append" mode the entire time
    # to write incrementally
    file_handle = open(output_file, "w", encoding="utf-8")
    file_handle.write(f"Start scraping: {start_url}\n\n")

    async def worker():
        try:
            while True:
                try:
                    url = await to_visit.get()
                except asyncio.CancelledError:
                    # Mark the task as done before exiting
                    if not to_visit.empty():
                        to_visit.task_done()
                    break

                # Wait here if user paused
                await is_paused_event.wait()

                if url in visited:
                    to_visit.task_done()
                    continue

                visited.add(url)
                logger.info(f"Scraping: {url}")

                async with semaphore:
                    html = await fetch_page(url, user_agent)

                if html:
                    # Convert HTML to text
                    text = html_to_markdown(html)
                    if text:
                        # Write incrementally to file
                        file_handle.write(f"URL: {url}\n")
                        file_handle.write(text)
                        file_handle.write("\n\n---\n\n")
                        file_handle.flush()
                        
                        # Store in ChromaDB if enabled
                        if db_handler:
                            db_handler.add_document(text, url)

                    # Extract new links
                    links = extract_links(html, url, domain, start_path)
                    for link in links:
                        if link not in visited:
                            await to_visit.put(link)

                    # Update progress
                    progress_bar.update(1)
                    update_progress()

                to_visit.task_done()
        except Exception as e:
            logger.error(f"Worker error: {e}")
            if not to_visit.empty():
                to_visit.task_done()
            raise

    # Create multiple workers
    tasks = []
    try:
        for _ in range(rate_limit):
            t = asyncio.create_task(worker())
            tasks.append(t)

        # Wait for the queue to empty
        await to_visit.join()
    finally:
        # Cancel all workers
        for t in tasks:
            t.cancel()
        # Wait for all tasks to complete their cancellation
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Close progress bar and file
        progress_bar.close()
        file_handle.write("Scraping completed.\n")
        file_handle.close()

        logger.info("Scraping completed.")

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
