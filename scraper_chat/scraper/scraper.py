"""
Web scraping module with support for:
- Recursive crawling with domain/path restrictions
- Proxy failover and rate limiting
- Robots.txt and sitemap.xml compliance
- GitHub repository scraping
- Content deduplication
- Markdown conversion
"""

import asyncio
import aiohttp
from typing import Optional, List, Dict, Any
from urllib.parse import urljoin, urlparse, urlunparse
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from urllib import robotparser
from trafilatura.core import extract_metadata
from trafilatura import extract
from tqdm import tqdm
import os
import base64
from ..text_processor import TextProcessor
from trafilatura.settings import use_config
from ..database.chroma_handler import ChromaHandler
from scraper_chat.logger.logging_config import logger
import re
from datetime import datetime

text_processor = TextProcessor()


def normalize_url(url: str) -> str:
    """
    Normalize URL by removing fragments and query parameters.

    Args:
        url: URL to normalize

    Returns:
        Normalized URL with only scheme, netloc, and path
    """
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def is_valid_url(url: str) -> bool:
    """
    Validate URL format and scheme.

    Args:
        url: URL to validate

    Returns:
        True if URL is valid and uses http(s) scheme
    """
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def get_domain(url: str) -> str:
    """
    Extract domain from URL.

    Args:
        url: URL to extract domain from

    Returns:
        Domain string or empty string if invalid
    """
    try:
        return urlparse(url).netloc
    except Exception:
        return ""


def is_same_domain(url1: str, url2: str) -> bool:
    """
    Check if two URLs belong to same domain.

    Args:
        url1: First URL
        url2: Second URL

    Returns:
        True if URLs have same domain
    """
    return get_domain(url1) == get_domain(url2)


def should_follow_link(url: str, base_domain: str, base_path: str) -> bool:
    """
    Check if link should be followed based on domain and path.

    Args:
        url: URL to check
        base_domain: Domain to restrict to
        base_path: Path prefix to restrict to

    Returns:
        True if link should be followed

    Note:
        base_domain should be normalized full URL
    """
    if not is_valid_url(url):
        return False

    if not is_same_domain(url, base_domain):
        return False

    path = urlparse(url).path
    return path.startswith(base_path)


def get_output_filename(url: str) -> str:
    """
    Generate filesystem-safe output filename from URL.

    Args:
        url: Source URL

    Returns:
        Path to output file

    Example:
        https://svelte.dev/docs/ -> svelte_dev.txt
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
    Remove fragment identifier from URL.

    Args:
        url: URL to process

    Returns:
        URL without fragment

    Example:
        http://example.com/page#section -> http://example.com/page
    """
    parsed = urlparse(url)
    parsed = parsed._replace(fragment="")
    return urlunparse(parsed)


async def get_next_proxy() -> Optional[str]:
    """
    Get current proxy from rotation.

    Returns:
        Proxy URL or None if no proxies configured

    Thread-safe via proxy_lock
    """
    async with proxy_lock:
        if not proxies_list:
            return None
        return proxies_list[current_proxy_index]


async def switch_to_next_proxy() -> None:
    """
    Rotate to next proxy in list.

    Thread-safe via proxy_lock.
    Wraps around to start if at end of list.
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
    Fetch page content with proxy failover.

    Args:
        url: URL to fetch
        user_agent: User agent string

    Returns:
        Page content as string

    Raises:
        RuntimeError: If all fetch attempts fail

    Features:
    - Automatic proxy rotation on failure
    - Retries for 403/429 status codes
    - Configurable timeout
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

        except aiohttp.ClientResponseError:
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
    Extract and filter in-domain links from HTML.

    Args:
        html: HTML content
        base_url: Base URL for resolving relative links
        domain: Domain to restrict links to
        start_path: Path prefix to restrict links to

    Returns:
        List of absolute URLs matching domain/path

    Features:
    - Tries trafilatura metadata first
    - Falls back to BeautifulSoup parsing
    - Resolves relative links
    - Removes duplicates
    - Filters by domain and path
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
    Convert HTML to Markdown with fallback.

    Args:
        html: HTML content

    Returns:
        Markdown text

    Features:
    - Uses trafilatura with custom config
    - Falls back to BeautifulSoup if trafilatura fails
    - Preserves formatting but removes links
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
    Fetch repository contents via GitHub API.

    Args:
        url: GitHub repository URL

    Returns:
        Dict containing:
        - owner: Repository owner
        - repo: Repository name
        - branch: Default branch
        - files: List of file objects

    Note:
        Requires GITHUB_TOKEN environment variable
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
    """
    Fetch single file from GitHub repository.

    Args:
        owner: Repository owner
        repo: Repository name
        path: File path in repository
        headers: GitHub API headers

    Returns:
        File content as string

    Raises:
        Exception: If file fetch fails
    """
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


async def check_sitemap(url: str, user_agent: str) -> Optional[List[str]]:
    """
    Find and parse sitemap.xml files.

    Args:
        url: Base URL
        user_agent: User agent string

    Returns:
        List of URLs from sitemap or None if not found

    Features:
    - Checks multiple common sitemap locations
    - Supports sitemap index files
    - Handles XML parsing errors gracefully
    """
    parsed = urlparse(url)
    root_domain = f"{parsed.scheme}://{parsed.netloc}"

    sitemap_locations = [
        urljoin(url, "sitemap.xml"),
        urljoin(root_domain, "sitemap.xml"),
        urljoin(root_domain, "sitemap_index.xml"),
    ]

    for sitemap_url in sitemap_locations:
        try:
            content = await fetch_page(sitemap_url, user_agent)
            if not content:
                continue

            try:
                root = ET.fromstring(content)
                urls = set()

                if "sitemapindex" in root.tag:
                    for sitemap in root.findall(".//{*}loc"):
                        sub_content = await fetch_page(sitemap.text, user_agent)
                        if sub_content:
                            try:
                                sub_root = ET.fromstring(sub_content)
                                urls.update(
                                    loc.text for loc in sub_root.findall(".//{*}loc")
                                )
                            except ET.ParseError:
                                continue
                else:
                    urls.update(loc.text for loc in root.findall(".//{*}loc"))

                if urls:
                    logger.info(f"Found {len(urls)} URLs in sitemap at {sitemap_url}")
                    return list(urls)

            except ET.ParseError:
                continue

        except Exception as e:
            logger.debug(f"Error checking sitemap at {sitemap_url}: {str(e)}")
            continue

    return None


async def check_robots_txt(
    url: str, user_agent: str
) -> Optional[robotparser.RobotFileParser]:
    """
    Fetch and parse robots.txt file.

    Args:
        url: Base URL
        user_agent: User agent string

    Returns:
        Configured RobotFileParser or None if not found
    """
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    try:
        content = await fetch_page(robots_url, user_agent)
        if content:
            parser = robotparser.RobotFileParser(robots_url)
            parser.parse(content.splitlines())
            logger.info(f"Found and parsed robots.txt at {robots_url}")
            return parser
    except Exception as e:
        logger.debug(f"Error fetching robots.txt at {robots_url}: {str(e)}")

    return None


def can_fetch(
    robots_parser: Optional[robotparser.RobotFileParser], url: str, user_agent: str
) -> bool:
    """
    Check if URL is allowed by robots.txt.

    Args:
        robots_parser: Configured parser or None
        url: URL to check
        user_agent: User agent string

    Returns:
        True if fetch is allowed or no parser exists
    """
    if robots_parser is None:
        return True
    return robots_parser.can_fetch(user_agent, url)


def is_blog_post(url: str, html: Optional[str] = None) -> bool:
    """
    Detect if a URL points to a blog post using various heuristics.

    Args:
        url: URL to check
        html: Optional HTML content for additional checks

    Returns:
        True if URL likely points to a blog post
    """
    url_lower = url.lower()
    path = urlparse(url).path.lower()

    # Check URL patterns indicating blog content
    blog_indicators = [
        "/blog/",
        "/posts/",
        "/news/",
        "/article/",
        "/updates/",
        "/weblog/",
    ]
    if any(indicator in url_lower for indicator in blog_indicators):
        return True

    # Check for date patterns in URL (e.g. /2024/01/21/, /2024-01-21)
    date_patterns = [
        r"/\d{4}/\d{2}/\d{2}/",  # /YYYY/MM/DD/
        r"/\d{4}-\d{2}-\d{2}",  # /YYYY-MM-DD
        r"/\d{4}/[a-z]+/\d{1,2}/",  # /YYYY/month/DD/
        r"/\d{4}/\d{1,2}/\d{1,2}",  # /YYYY/M/D
        r"/\d{4}/[a-z]{3}/\d{2}/",  # /YYYY/feb/DD/
    ]
    if any(re.search(pattern, url) for pattern in date_patterns):
        return True

    # If HTML content is available, check metadata
    if html:
        try:
            metadata = extract_metadata(html)
            if metadata:
                # Check if article type indicates blog
                article_type = metadata.get("type", "").lower()
                if article_type in ["blog", "article", "post"]:
                    return True

                # Check publication date - if recent, likely a blog post
                pub_date = metadata.get("date")
                if pub_date:
                    try:
                        date = datetime.fromisoformat(pub_date)
                        # Consider it a blog if published in last 2 years
                        if (datetime.now() - date).days < 730:
                            return True
                    except ValueError:
                        pass
        except Exception as e:
            logger.debug(f"Error checking metadata for blog detection: {e}")

    return False


async def scrape_recursive(
    start_url: str,
    user_agent: str,
    rate_limit: int,
    dump_text: bool = False,
    force_rescrape: bool = False,
) -> str:
    """
    Recursively scrape website with constraints.

    Args:
        start_url: Starting URL
        user_agent: User agent string
        rate_limit: Max concurrent requests
        dump_text: Save raw text files
        force_rescrape: Ignore existing content

    Returns:
        Status message

    Features:
    - GitHub repository support
    - Robots.txt compliance
    - Sitemap.xml support
    - Rate limiting
    - Progress tracking
    - Content deduplication
    - Proxy failover
    - Blog post filtering
    """
    if not start_url.startswith(("http://", "https://")):
        return "Error: Invalid URL. Must start with http:// or https://"

    if "github.com" in start_url and "/blob/" not in start_url:
        logger.info(f"Detected GitHub repository URL: {start_url}")
        db_handler: Optional[ChromaHandler] = ChromaHandler(
            ChromaHandler.get_collection_name(start_url)
        )
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
    db_handler: ChromaHandler = ChromaHandler(
        ChromaHandler.get_collection_name(start_url)
    )

    # Check robots.txt first
    robots_parser = await check_robots_txt(start_url, user_agent)
    if robots_parser and not can_fetch(robots_parser, start_url, user_agent):
        return "Error: URL is disallowed by robots.txt"

    # Check for sitemap
    sitemap_urls = await check_sitemap(start_url, user_agent)
    if sitemap_urls:
        logger.info(f"Found {len(sitemap_urls)} URLs in sitemap")
        # Filter out blog posts from sitemap
        non_blog_urls = [url for url in sitemap_urls if not is_blog_post(url)]
        logger.info(f"Filtered to {len(non_blog_urls)} non-blog URLs")
        
        # Queue all valid non-blog URLs for scraping
        queued_count = 0
        for url in non_blog_urls:
            normalized_url = remove_anchor(url)
            if normalized_url not in visited and should_follow_link(normalized_url, start_url, start_path):
                await to_visit.put(normalized_url)
                queued_count += 1
                logger.info(f"Queued {normalized_url} for scraping")
        
        logger.info(f"Queued {queued_count} URLs for scraping")
    else:
        logger.info("No sitemap found, using recursive crawling")
        await to_visit.put(remove_anchor(start_url))

    semaphore = asyncio.Semaphore(rate_limit)
    progress_bar = tqdm(desc="Pages Scraped", unit="pages", total=0)

    def update_progress() -> None:
        logger.info(
            f"Pages Scraped: {progress_bar.n} pages [{progress_bar.format_dict.get('rate', 0):.2f} pages/s]"
        )

    file_handle = None
    if dump_text:
        file_handle = open(output_file, "w", encoding="utf-8")
        file_handle.write(f"Start scraping: {start_url}\n\n")

    async def worker() -> None:
        try:
            while True:
                try:
                    url = await to_visit.get()
                    try:
                        if url in visited:
                            continue

                        # Check robots.txt before processing URL
                        if not can_fetch(robots_parser, url, user_agent):
                            logger.info(f"Skipping {url} - disallowed by robots.txt")
                            continue

                        visited.add(url)
                        logger.info(f"Scraping: {url}")
                        async with semaphore:
                            html = await fetch_page(url, user_agent)
                            if html:
                                text = html_to_markdown(html)
                                if text:
                                    # Check if we should skip this URL due to duplicate content
                                    if (
                                        not force_rescrape
                                        and db_handler.has_matching_content(url, text)
                                    ):
                                        logger.info(
                                            f"Skipping duplicate content: {url}"
                                        )
                                        continue

                                    # Always store in database
                                    db_handler.add_document(
                                        text, url, force_rescrape=force_rescrape
                                    )

                                    # Optionally save raw text files
                                    if dump_text:
                                        file_handle.write(
                                            f"URL: {url}\n{text}\n\n---\n\n"
                                        )
                                        file_handle.flush()

                                    # Extract and queue new links
                                    links = extract_links(html, url, domain, start_path)
                                    for link in links:
                                        if link not in visited and can_fetch(
                                            robots_parser, link, user_agent
                                        ):
                                            await to_visit.put(link)

                        progress_bar.update(1)
                        update_progress()
                    finally:
                        to_visit.task_done()
                except Exception as e:
                    logger.error(f"Error processing {url}: {str(e)}")
                    to_visit.task_done()
        except asyncio.CancelledError:
            logger.debug("Worker cancelled")
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
        if dump_text:
            file_handle.write("Scraping completed.\n")
            file_handle.close()
        logger.info("Scraping completed.")

    return "Scraping completed successfully."
