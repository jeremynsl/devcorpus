import asyncio
import aiohttp
from urllib.parse import urljoin, urlparse, urlunparse
from bs4 import BeautifulSoup
from trafilatura.core import extract_metadata
from trafilatura import extract
from ..text_processor import TextProcessor
from tqdm import tqdm
from trafilatura.settings import use_config
from ..database.chroma_handler import ChromaHandler
from scraper_chat.logger.logging_config import logger
import os
import sys
import logging
import base64

text_processor = TextProcessor()


def normalize_url(url: str) -> str:
    """Normalize URL by removing fragments and query parameters"""
    parsed = urlparse(url)
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            "",  # params
            "",  # query
            "",  # fragment
        )
    )


def is_valid_url(url: str) -> bool:
    """Check if URL is valid and uses http(s) scheme"""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except:
        return False


def get_domain(url: str) -> str:
    """Extract domain from URL"""
    try:
        return urlparse(url).netloc
    except:
        return ""


def is_same_domain(url1: str, url2: str) -> bool:
    """Check if two URLs belong to the same domain"""
    return get_domain(url1) == get_domain(url2)


def should_follow_link(url: str, base_domain: str, base_path: str) -> bool:
    """Determine if link should be followed based on domain and path restrictions"""
    if not is_valid_url(url):
        return False

    # Must be same domain
    if not is_same_domain(url, base_domain):
        return False

    # Must start with base path
    path = urlparse(url).path
    return path.startswith(base_path)


def get_output_filename(url: str) -> str:
    """
    Convert URL to a valid filename by removing special characters and slashes.
    Example: https://svelte.dev/docs/ becomes svelte_dev.txt
    """
    parsed = urlparse(url)

    # Get project root directory (two levels up from this file)
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_dir = os.path.join(root_dir, "scrapedtxt")
    os.makedirs(output_dir, exist_ok=True)

    # Use domain name as filename, removing special characters
    filename = parsed.netloc.replace(".", "_").replace("-", "_")
    if not filename:
        filename = "default"
    return os.path.join(output_dir, f"{filename}.txt")


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

        # Prepare session initialization arguments
        session_args = {
            "headers": {"User-Agent": user_agent},
            "timeout": aiohttp.ClientTimeout(total=30),
        }
        # If a proxy URL is available, add it to the arguments
        if proxy_url:
            session_args["proxy"] = proxy_url

        try:
            # Create session
            async with aiohttp.ClientSession(**session_args) as session:
                # Fetch the page
                response = await session.get(url)

                # Attempt to get text content
                try:
                    content = await response.text()
                except Exception:
                    # If text extraction fails, switch proxy
                    await switch_to_next_proxy()
                    if attempts < max_retries:
                        continue
                    raise

                # Check for HTTP errors
                if response.status in (403, 429):
                    logger.warning(
                        f"Received status {response.status} for {url}. "
                        f"Switching proxy and retrying. (Attempt {attempts}/{max_retries})"
                    )
                    await switch_to_next_proxy()
                    if attempts < max_retries:
                        continue
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                    )

                # Raise for other HTTP errors
                response.raise_for_status()

                # Return content
                return content

        except aiohttp.ClientResponseError as e:
            logger.warning(
                f"Received error for {url}. "
                f"Switching proxy and retrying. (Attempt {attempts}/{max_retries})"
            )
            await switch_to_next_proxy()
            if attempts < max_retries:
                continue
            raise

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"Error fetching {url}: {e}")
            await switch_to_next_proxy()
            if attempts < max_retries:
                continue
            raise

    # If all attempts fail
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts")


def extract_links(html: str, base_url: str, domain: str, start_path: str) -> list[str]:
    """
    Extract all in-domain links from the HTML.

    Args:
        html: HTML content to parse
        base_url: URL of the current page
        domain: Target domain to match (from urlparse(start_url).netloc)
        start_path: Base path to constrain crawling (from urlparse(start_url).path)
    """
    filtered_links = []

    # Ensure start_path ends with slash for prefix matching
    if not start_path.endswith("/"):
        start_path += "/"

    # Try trafilatura first
    try:
        metadata = extract_metadata(html, default_url=base_url)
        if metadata:
            links = metadata.as_dict().get("links", [])
            if links:
                # Handle nested lists from some trafilatura versions
                if isinstance(links[0], list):
                    links = [item for sublist in links for item in sublist]

                for link in links:
                    try:
                        absolute_link = urljoin(base_url, link)
                        link_no_anchor = remove_anchor(absolute_link)
                        parsed_link = urlparse(link_no_anchor)

                        # Validate both domain and path constraints
                        if parsed_link.netloc == domain and parsed_link.path.startswith(
                            start_path
                        ):
                            filtered_links.append(link_no_anchor)
                    except Exception as e:
                        logger.error(f"Error processing link {link}: {str(e)}")
                        continue

                return list(set(filtered_links))  # Deduplicate
    except Exception as e:
        logger.debug(f"Trafilatura link extraction failed: {str(e)}")

    # Fallback to BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for a_tag in soup.find_all("a", href=True):
        try:
            raw_link = a_tag["href"]
            absolute_link = urljoin(base_url, raw_link)
            link_no_anchor = remove_anchor(absolute_link)
            parsed_link = urlparse(link_no_anchor)

            if parsed_link.netloc == domain and parsed_link.path.startswith(start_path):
                filtered_links.append(link_no_anchor)
        except Exception as e:
            logger.error(f"Error processing tag {a_tag}: {str(e)}")
            continue

    return list(set(filtered_links))  # Deduplicate


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


async def fetch_github_content(url: str) -> dict:
    """
    Fetch content from a GitHub repository using the GitHub API.
    Returns repository info including default branch and tree.
    """
    path = urlparse(url).path.strip("/")
    owner, repo = path.split("/")[:2]

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "GitHubScraper",
        "Authorization": f"token {os.getenv('GITHUB_TOKEN')}",
    }

    async with aiohttp.ClientSession() as session:
        # Get repository info and default branch
        async with session.get(
            f"https://api.github.com/repos/{owner}/{repo}", headers=headers
        ) as response:
            if response.status == 200:
                repo_info = await response.json()
                branch = repo_info["default_branch"]
            else:
                raise Exception(f"Failed to get repository info: {response.status}")

        # Get repository tree
        async with session.get(
            f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1",
            headers=headers,
        ) as response:
            if response.status == 200:
                tree = await response.json()
                return {
                    "owner": owner,
                    "repo": repo,
                    "branch": branch,
                    "files": [
                        f
                        for f in tree["tree"]
                        if f["type"] == "blob"
                        and f["path"].endswith(
                            (
                                ".md",
                                ".js",
                                ".ts",
                                ".py",
                                ".jsx",
                                ".tsx",
                                ".vue",
                                ".svelte",
                                ".html",
                                ".css",
                            )
                        )
                    ],
                }
            else:
                raise Exception(f"Failed to get repository tree: {response.status}")


async def fetch_github_file(owner: str, repo: str, path: str, headers: dict) -> str:
    """Fetch a single file's content from GitHub."""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://api.github.com/repos/{owner}/{repo}/contents/{path}",
            headers=headers,
        ) as response:
            if response.status == 200:
                data = await response.json()
                return base64.b64decode(data["content"]).decode("utf-8")
            else:
                raise Exception(f"Failed to get file content: {response.status}")


async def scrape_recursive(
    start_url: str, user_agent: str, rate_limit: int, use_db: bool = False
):
    """
    Recursively scrape all links from start_url domain.
    Writes results incrementally to a file in the scrapedtxt directory.
    If use_db is True, also stores content in ChromaDB.
    """
    if not start_url.startswith(("http://", "https://")):
        return "Error: Invalid URL. Must start with http:// or https://"

    else:
        # Check if it's a GitHub repository
        if "github.com" in start_url and "/blob/" not in start_url:
            logger.info(f"Detected GitHub repository URL: {start_url}")

            # Initialize ChromaDB if requested
            db_handler = None
            if use_db:
                collection_name = ChromaHandler.get_collection_name(start_url)
                db_handler = ChromaHandler(collection_name)

            # Get repository content
            repo_info = await fetch_github_content(start_url)

            # Create a progress bar for GitHub files
            progress_bar = tqdm(
                desc="Files Processed", unit="files", total=len(repo_info["files"])
            )

            # Process each file
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "GitHubScraper",
                "Authorization": f"token {os.getenv('GITHUB_TOKEN')}",
            }

            for file in repo_info["files"]:
                try:
                    content = await fetch_github_file(
                        repo_info["owner"], repo_info["repo"], file["path"], headers
                    )

                    if content:
                        file_url = (
                            f"{start_url}/blob/{repo_info['branch']}/{file['path']}"
                        )

                        # Store in ChromaDB if enabled
                        if db_handler:
                            db_handler.add_document(content, file_url)

                        # Update progress
                        progress_bar.update(1)
                        logger.info(f"Processed: {file['path']}")

                except Exception as e:
                    logger.error(f"Error processing {file['path']}: {str(e)}")
                    continue

            progress_bar.close()
            return f"Successfully processed {len(repo_info['files'])} files from GitHub repository"

        # Regular website scraping
        output_file = get_output_filename(start_url)
        domain = urlparse(start_url).netloc

        start_path = urlparse(start_url).path
        if not start_path:
            start_path = "/"

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
            logger.info(
                f"Pages Scraped: {progress_bar.n} pages [{progress_bar.format_dict['rate']:.2f} pages/s]"
            )

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
                        raise  # Re-raise to properly handle cancellation
                    except asyncio.QueueEmpty:
                        break

                    try:
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
                                    logging.debug(f"Storedin ChromaDB: {url}")

                            # Extract new links
                            links = extract_links(html, url, domain, start_path)
                            logging.debug(f"Extracted links: {links}")
                            for link in links:
                                if link not in visited:
                                    await to_visit.put(link)

                            # Update progress
                            progress_bar.update(1)
                            update_progress()

                        to_visit.task_done()
                    except Exception as e:
                        logger.error(f"Error processing {url}: {e}")
                        to_visit.task_done()  # Always mark task as done even on error
                        continue  # Continue to next URL instead of crashing worker
            except asyncio.CancelledError:
                # Handle final cleanup on cancellation
                logger.info("Worker cancelled, cleaning up...")
                raise
            except Exception as e:
                logger.error(f"Worker error: {e}")
                raise

        # Create multiple workers
        tasks = []
        try:
            for _ in range(rate_limit):
                t = asyncio.create_task(worker())
                tasks.append(t)

            # Wait for the queue to empty
            await to_visit.join()
        except Exception as e:
            logger.error(f"Error during scraping: {e}")
            raise
        finally:
            # Cancel all workers
            for t in tasks:
                if not t.done():
                    t.cancel()

            # Wait for all tasks to complete their cancellation
            if tasks:
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                except Exception as e:
                    logger.error(f"Error during worker cleanup: {e}")

            # Close progress bar and file
            progress_bar.close()
            file_handle.write("Scraping completed.\n")
            file_handle.close()

            logger.info("Scraping completed.")
