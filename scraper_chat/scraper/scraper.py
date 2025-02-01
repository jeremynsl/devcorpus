import asyncio
import aiohttp
from typing import Optional, List, Dict, Any
from urllib.parse import urljoin, urlparse, urlunparse
from bs4 import BeautifulSoup
from trafilatura.core import extract_metadata
from trafilatura import extract
from tqdm import tqdm
import os
import base64
from ..text_processor import TextProcessor
from trafilatura.settings import use_config
from ..database.chroma_handler import ChromaHandler
from scraper_chat.logger.logging_config import logger

text_processor = TextProcessor()


def normalize_url(url: str) -> str:
    """Normalize URL by removing fragments and query parameters."""
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def is_valid_url(url: str) -> bool:
    """Check if URL is valid and uses http(s) scheme."""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:  # Consider catching a more specific exception
        return False


def get_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        return urlparse(url).netloc
    except Exception:
        return ""


def is_same_domain(url1: str, url2: str) -> bool:
    """Check if two URLs belong to the same domain."""
    return get_domain(url1) == get_domain(url2)


def should_follow_link(url: str, base_domain: str, base_path: str) -> bool:
    """
    Determine if link should be followed based on domain and path restrictions.

    Note: `base_domain` should be a full URL (or ensure it is normalized).
    """
    if not is_valid_url(url):
        return False

    if not is_same_domain(url, base_domain):
        return False

    path = urlparse(url).path
    return path.startswith(base_path)


def get_output_filename(url: str) -> str:
    """
    Convert URL to a valid filename by removing special characters and slashes.
    Example: https://svelte.dev/docs/ becomes svelte_dev.txt
    """
    parsed = urlparse(url)
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_dir = os.path.join(root_dir, "scrapedtxt")
    os.makedirs(output_dir, exist_ok=True)

    filename = parsed.netloc.replace(".", "_").replace("-", "_")
    if not filename:
        filename = "default"
    return os.path.join(output_dir, f"{filename}.txt")


###############################################################################
# Proxy Failover
###############################################################################
proxy_lock = asyncio.Lock()
current_proxy_index: int = 0
proxies_list: List[str] = []  # Will be populated after loading config


def remove_anchor(url: str) -> str:
    """
    Remove the anchor/fragment from a URL so that
    http://example.com/page#something == http://example.com/page
    """
    parsed = urlparse(url)
    parsed = parsed._replace(fragment="")
    return urlunparse(parsed)


async def get_next_proxy() -> Optional[str]:
    """
    Returns the currently selected proxy from the proxies_list.
    If none exist, return None.
    """
    async with proxy_lock:
        if not proxies_list:
            return None
        return proxies_list[current_proxy_index]


async def switch_to_next_proxy() -> None:
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
    max_retries: int = max(1, len(proxies_list) or 1)
    attempts: int = 0

    while attempts < max_retries:
        attempts += 1
        proxy_url: Optional[str] = await get_next_proxy()
        logger.debug(f"Fetching {url} [Attempt {attempts}] (Proxy={proxy_url})")

        session_args: Dict[str, Any] = {
            "headers": {"User-Agent": user_agent},
            "timeout": aiohttp.ClientTimeout(total=30),
        }
        if proxy_url:
            session_args["proxy"] = proxy_url

        try:
            async with aiohttp.ClientSession(**session_args) as session:
                response = await session.get(url)
                try:
                    content: str = await response.text()
                except Exception:
                    await switch_to_next_proxy()
                    if attempts < max_retries:
                        continue
                    raise

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

                response.raise_for_status()
                return content

        except aiohttp.ClientResponseError as e:
            logger.warning(
                f"Received error for {url}. Switching proxy and retrying. (Attempt {attempts}/{max_retries})"
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

    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts")


def extract_links(html: str, base_url: str, domain: str, start_path: str) -> List[str]:
    """
    Extract all in-domain links from the HTML.
    """
    filtered_links: List[str] = []

    if not start_path.endswith("/"):
        start_path += "/"

    try:
        metadata = extract_metadata(html, default_url=base_url)
        if metadata:
            links = metadata.as_dict().get("links", [])
            if links:
                if isinstance(links[0], list):
                    links = [item for sublist in links for item in sublist]

                for link in links:
                    try:
                        absolute_link = urljoin(base_url, link)
                        link_no_anchor = remove_anchor(absolute_link)
                        parsed_link = urlparse(link_no_anchor)

                        if parsed_link.netloc == domain and parsed_link.path.startswith(
                            start_path
                        ):
                            filtered_links.append(link_no_anchor)
                    except Exception as e:
                        logger.error(f"Error processing link {link}: {str(e)}")
                        continue

                return list(set(filtered_links))
    except Exception as e:
        logger.debug(f"Trafilatura link extraction failed: {str(e)}")

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

    return list(set(filtered_links))


def html_to_markdown(html: str) -> str:
    """
    Convert HTML to Markdown text using trafilatura.
    """
    cleaned_html: str = text_processor.preprocess_html(html)
    config = use_config()
    config.set("DEFAULT", "include_links", "False")
    config.set("DEFAULT", "include_formatting", "True")
    config.set("DEFAULT", "output_format", "markdown")
    config.set("DEFAULT", "extraction_timeout", "0")

    text: Optional[str] = extract(cleaned_html, config=config)
    if not text:
        logger.warning("Trafilatura extraction failed, falling back to BeautifulSoup")
        soup = BeautifulSoup(cleaned_html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)

    return text or ""


async def fetch_github_content(url: str) -> Dict[str, Any]:
    """
    Fetch content from a GitHub repository using the GitHub API.
    """
    path = urlparse(url).path.strip("/")
    owner, repo = path.split("/")[:2]

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "GitHubScraper",
        "Authorization": f"token {os.getenv('GITHUB_TOKEN')}",
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://api.github.com/repos/{owner}/{repo}", headers=headers
        ) as response:
            if response.status == 200:
                repo_info = await response.json()
                branch = repo_info["default_branch"]
            else:
                raise Exception(f"Failed to get repository info: {response.status}")

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


async def fetch_github_file(
    owner: str, repo: str, path: str, headers: Dict[str, str]
) -> str:
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
) -> str:
    """
    Recursively scrape all links from start_url domain.
    """
    if not start_url.startswith(("http://", "https://")):
        return "Error: Invalid URL. Must start with http:// or https://"

    if "github.com" in start_url and "/blob/" not in start_url:
        logger.info(f"Detected GitHub repository URL: {start_url}")
        db_handler: Optional[ChromaHandler] = None
        if use_db:
            collection_name = ChromaHandler.get_collection_name(start_url)
            db_handler = ChromaHandler(collection_name)
        repo_info = await fetch_github_content(start_url)
        progress_bar = tqdm(
            desc="Files Processed", unit="files", total=len(repo_info["files"])
        )
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
                    file_url = f"{start_url}/blob/{repo_info['branch']}/{file['path']}"
                    if db_handler:
                        db_handler.add_document(content, file_url)
                    progress_bar.update(1)
                    logger.info(f"Processed: {file['path']}")
            except Exception as e:
                logger.error(f"Error processing {file['path']}: {str(e)}")
                continue
        progress_bar.close()
        return f"Successfully processed {len(repo_info['files'])} files from GitHub repository"

    output_file = get_output_filename(start_url)
    domain: str = urlparse(start_url).netloc
    start_path: str = urlparse(start_url).path or "/"
    visited: set[str] = set()
    to_visit: asyncio.Queue[str] = asyncio.Queue()
    db_handler: Optional[ChromaHandler] = None
    if use_db:
        collection_name = ChromaHandler.get_collection_name(start_url)
        db_handler = ChromaHandler(collection_name)
    await to_visit.put(remove_anchor(start_url))
    semaphore = asyncio.Semaphore(rate_limit)
    progress_bar = tqdm(desc="Pages Scraped", unit="pages", total=0)

    def update_progress() -> None:
        logger.info(
            f"Pages Scraped: {progress_bar.n} pages [{progress_bar.format_dict.get('rate', 0):.2f} pages/s]"
        )

    file_handle = open(output_file, "w", encoding="utf-8")
    file_handle.write(f"Start scraping: {start_url}\n\n")

    async def worker() -> None:
        try:
            while True:
                try:
                    url = await to_visit.get()
                except asyncio.CancelledError:
                    if not to_visit.empty():
                        to_visit.task_done()
                    raise
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
                        text = html_to_markdown(html)
                        if text:
                            file_handle.write(f"URL: {url}\n{text}\n\n---\n\n")
                            file_handle.flush()
                            if db_handler:
                                db_handler.add_document(text, url)
                                logger.debug(f"Stored in ChromaDB: {url}")
                        links = extract_links(html, url, domain, start_path)
                        logger.debug(f"Extracted links: {links}")
                        for link in links:
                            if link not in visited:
                                await to_visit.put(link)
                        progress_bar.update(1)
                        update_progress()
                    to_visit.task_done()
                except Exception as e:
                    logger.error(f"Error processing {url}: {e}")
                    to_visit.task_done()
                    continue
        except asyncio.CancelledError:
            logger.info("Worker cancelled, cleaning up...")
            raise
        except Exception as e:
            logger.error(f"Worker error: {e}")
            raise

    tasks: List[asyncio.Task] = []
    try:
        for _ in range(rate_limit):
            t = asyncio.create_task(worker())
            tasks.append(t)
        await to_visit.join()
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        raise
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error during worker cleanup: {e}")
        progress_bar.close()
        file_handle.write("Scraping completed.\n")
        file_handle.close()
        logger.info("Scraping completed.")

    return "Scraping completed successfully."
