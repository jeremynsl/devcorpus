import pytest
from unittest.mock import patch, AsyncMock, mock_open, MagicMock
import sys
from pathlib import Path
import asyncio

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from scraper_chat.scraper.scraper import (
    scrape_recursive,
    fetch_page,
    extract_links,
    fetch_github_file,
    remove_anchor,
    get_output_filename,
    fetch_github_content,
)


@pytest.fixture
def mock_config():
    """Create a mock config"""
    return {"proxies": [], "rate_limit": 2, "user_agent": "TestBot/1.0"}


@pytest.fixture
def mock_html():
    """Create mock HTML content"""
    return """
    <html>
        <body>
            <h1>Test Page</h1>
            <p>Some test content</p>
            <a href="/page1">Internal Link 1</a>
            <a href="/page2">Internal Link 2</a>
            <a href="/page3#anchor">Internal Link with Anchor</a>
        </body>
    </html>
    """


@pytest.fixture
def mock_proxies():
    """Mock global proxies list"""
    import scraper_chat.scraper.scraper

    scraper_chat.scraper.scraper.proxies_list = []
    return scraper_chat.scraper.scraper.proxies_list


@pytest.fixture
def mock_github_response():
    """Create mock GitHub API responses"""
    repo_info = {"default_branch": "main"}

    tree = {
        "tree": [
            {"path": "src/main.py", "type": "blob"},
            {"path": "README.md", "type": "blob"},
            {"path": "tests/test_main.py", "type": "blob"},
            {"path": "docs/index.html", "type": "blob"},
            {"path": ".gitignore", "type": "blob"},  # Should be filtered out
        ]
    }

    file_content = {
        "content": "SGVsbG8gV29ybGQ="  # base64 encoded "Hello World"
    }

    return repo_info, tree, file_content


def test_get_output_filename():
    """Test URL to filename conversion"""
    import os

    # Get the expected base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scrapedtxt_dir = os.path.join(base_dir, "scrapedtxt")

    test_cases = [
        ("https://svelte.dev/docs/", os.path.join(scrapedtxt_dir, "svelte_dev.txt")),
        ("http://example.com/path", os.path.join(scrapedtxt_dir, "example_com.txt")),
        ("https://test.org/path", os.path.join(scrapedtxt_dir, "test_org.txt")),
    ]

    for url, expected in test_cases:
        result = get_output_filename(url)
        assert result == expected, f"Expected {expected}, got {result}"


def test_get_output_filename_edge_cases():
    """Test get_output_filename with edge case URLs"""
    import os

    # Get the expected base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scrapedtxt_dir = os.path.join(base_dir, "scrapedtxt")

    test_cases = [
        ("", os.path.join(scrapedtxt_dir, "default.txt")),
        ("http://", os.path.join(scrapedtxt_dir, "default.txt")),
        ("https://", os.path.join(scrapedtxt_dir, "default.txt")),
    ]

    for url, expected in test_cases:
        result = get_output_filename(url)
        assert result == expected, f"Expected {expected}, got {result}"


def test_remove_anchor():
    """Test anchor removal from URLs"""
    test_cases = [
        ("http://example.com/page", "http://example.com/page"),
        ("http://example.com/page#section", "http://example.com/page"),
        ("http://example.com/page#", "http://example.com/page"),
    ]
    for url, expected in test_cases:
        assert remove_anchor(url) == expected


@pytest.mark.asyncio
async def test_fetch_page():
    """Test page fetching with retries"""
    # Create a mock response that mimics an aiohttp response
    mock_response = AsyncMock()
    mock_response.text = AsyncMock(return_value="Test content")
    mock_response.raise_for_status = AsyncMock()

    # Create a mock session
    mock_session = AsyncMock()
    # Setup the session so that its async context manager returns itself
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = None

    # Two different ways to mock get method to cover both scenarios
    def mock_get_awaitable(url, **kwargs):
        return mock_response

    # Reset mocks
    mock_response.reset_mock()
    mock_session.reset_mock()

    # Patch the session's get method
    mock_session.get = AsyncMock(side_effect=mock_get_awaitable)

    # Patch aiohttp.ClientSession to use our mock_session
    with patch("aiohttp.ClientSession", return_value=mock_session):
        content = await fetch_page("http://test.com", "TestBot/1.0")

        # Verify content
        assert content == "Test content"

        # Verify method was called
        mock_session.get.assert_called_once_with("http://test.com")
        mock_response.raise_for_status.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_page_proxy_failover():
    """Test fetch_page with proxy failover"""
    import aiohttp
    import scraper_chat.scraper.scraper

    # Prepare proxy list
    scraper_chat.scraper.scraper.proxies_list = [
        "http://proxy1.com",
        "http://proxy2.com",
    ]

    # Create a mock session that handles both async context and method calls
    class MockClientSession:
        get_calls = 0  # Class-level counter

        def __init__(self, *args, **kwargs):
            self.proxy = kwargs.get("proxy")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, **kwargs):
            MockClientSession.get_calls += 1

            # Simulate first call failing, second call succeeding
            if MockClientSession.get_calls == 1:
                # Create a mock response with 403 status
                class MockFailResponse:
                    status = 403
                    request_info = None
                    history = None

                    async def text(self):
                        raise aiohttp.ClientResponseError(
                            request_info=None, history=None, status=403
                        )

                    def raise_for_status(self):
                        raise aiohttp.ClientResponseError(
                            request_info=None, history=None, status=403
                        )

                return MockFailResponse()

            # Create a mock response object
            class MockResponse:
                status = 200
                request_info = None
                history = None

                async def text(self):
                    return "Success content"

                def raise_for_status(self):
                    pass

            return MockResponse()

    # Patch aiohttp.ClientSession to use our mock
    with (
        patch("aiohttp.ClientSession", MockClientSession),
        patch(
            "scraper_chat.scraper.scraper.switch_to_next_proxy", new_callable=AsyncMock
        ) as mock_switch_proxy,
    ):
        # Fetch page
        content = await scraper_chat.scraper.scraper.fetch_page(
            "http://test.com", "TestBot/1.0"
        )
        assert content == "Success content"
        assert mock_switch_proxy.call_count == 1


def test_extract_links(mock_html):
    """Test link extraction"""
    # Mock trafilatura metadata
    mock_metadata = MagicMock()
    mock_metadata.as_dict.return_value = {
        "links": [
            "http://example.com/start/page1",
            "http://example.com/start/page2",
            "http://example.com/start/page3",
        ]
    }

    with patch(
        "scraper_chat.scraper.scraper.extract_metadata", return_value=mock_metadata
    ):
        links = extract_links(
            mock_html, "http://example.com/start", "example.com", "/start/"
        )
        assert len(links) == 3
        assert all(link.startswith("http://example.com/start/") for link in links)


def test_extract_links_empty_metadata():
    """Test extract_links with empty metadata"""
    import scraper_chat.scraper.scraper

    # Prepare mock HTML and metadata
    test_html = "<html><body><a href='/test'>Test Link</a></body></html>"
    base_url = "http://example.com"

    # Mock metadata extraction to return None
    with (
        patch("scraper_chat.scraper.scraper.extract_metadata", return_value=None),
        patch("scraper_chat.scraper.scraper.BeautifulSoup") as MockSoup,
    ):
        # Setup mock BeautifulSoup
        mock_soup = MagicMock()
        a_tag = MagicMock()
        a_tag.__getitem__.return_value = "/test"
        mock_soup.find_all.return_value = [a_tag]
        MockSoup.return_value = mock_soup

        # Extract links
        links = scraper_chat.scraper.scraper.extract_links(
            test_html, base_url, "example.com", "/"
        )

        # Verify links
        assert len(links) == 1
        assert links[0] == "http://example.com/test"


def test_extract_links_link_processing_error():
    """Test extract_links with link processing errors"""
    import scraper_chat.scraper.scraper

    # Prepare mock HTML with problematic links
    test_html = """
    <html>
        <body>
            <a href='javascript:void(0)'>Invalid Link</a>
            <a href='#'>Anchor Link</a>
            <a href='/valid'>Valid Link</a>
        </body>
    </html>
    """
    base_url = "http://example.com"

    # Patch BeautifulSoup to simulate link processing errors
    with (
        patch("scraper_chat.scraper.scraper.BeautifulSoup") as MockSoup,
        patch("scraper_chat.scraper.scraper.logger.error") as mock_error,
        patch("scraper_chat.scraper.scraper.extract_metadata", return_value=None),
    ):
        # Create mock a_tags with different problematic scenarios
        def mock_find_all(tag, **kwargs):
            mock_tags = [
                MagicMock(get=lambda x: "javascript:void(0)"),
                MagicMock(get=lambda x: "#"),
                MagicMock(get=lambda x: "/valid"),
            ]
            return mock_tags

        # Setup mock BeautifulSoup
        mock_soup = MagicMock()
        mock_soup.find_all.side_effect = mock_find_all
        MockSoup.return_value = mock_soup

        # Simulate link processing errors for first two links
        with patch(
            "scraper_chat.scraper.scraper.urljoin",
            side_effect=[
                Exception("Invalid link"),
                Exception("Anchor link"),
                "http://example.com/valid",
            ],
        ):
            links = scraper_chat.scraper.scraper.extract_links(
                test_html, base_url, "example.com", "/"
            )

            # Verify links
            assert len(links) == 1
            assert links[0] == "http://example.com/valid"

            # Verify error logging for problematic links
            assert mock_error.call_count >= 2


@pytest.mark.asyncio
async def test_scrape_recursive():
    """Test recursive scraping"""

    async def mock_fetch(*args, **kwargs):
        return """
            <html><body>
                <h1>Test Page</h1>
                <p>Test content</p>
                <a href="/page2">Link</a>
            </body></html>
        """

    m = mock_open()
    with (
        patch("builtins.open", m),
        patch("scraper_chat.scraper.scraper.fetch_page", side_effect=mock_fetch),
        patch("scraper_chat.scraper.scraper.tqdm"),
    ):

        async def run_with_timeout():
            try:
                async with asyncio.timeout(2):
                    await scrape_recursive(
                        "http://test.com", "TestBot/1.0", rate_limit=2, dump_text=False
                    )
            except asyncio.TimeoutError:
                # Get all tasks that are still running
                tasks = [
                    t for t in asyncio.all_tasks() if t is not asyncio.current_task()
                ]
                # Cancel them
                for task in tasks:
                    task.cancel()
                # Wait for all tasks to complete their cancellation
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

        await run_with_timeout()
        assert m.called


@pytest.mark.asyncio
async def test_fetch_github_content(mock_github_response):
    """Test fetching GitHub repository content"""
    repo_info, tree, _ = mock_github_response

    # Create mock responses for both API calls
    mock_repo_response = AsyncMock()
    mock_repo_response.status = 200
    mock_repo_response.json = AsyncMock(return_value=repo_info)

    mock_tree_response = AsyncMock()
    mock_tree_response.status = 200
    mock_tree_response.json = AsyncMock(return_value=tree)

    # Create mock context managers for session.get()
    mock_repo_context = AsyncMock()
    mock_repo_context.__aenter__ = AsyncMock(return_value=mock_repo_response)
    mock_repo_context.__aexit__ = AsyncMock(return_value=None)

    mock_tree_context = AsyncMock()
    mock_tree_context.__aenter__ = AsyncMock(return_value=mock_tree_response)
    mock_tree_context.__aexit__ = AsyncMock(return_value=None)

    # Create a mock session
    mock_session = AsyncMock()
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = None

    # Mock the get method to return appropriate context based on URL
    def mock_get(url, headers):
        if "trees" in url:
            return mock_tree_context
        return mock_repo_context

    mock_session.get = MagicMock(side_effect=mock_get)

    # Patch aiohttp.ClientSession to use our mock_session
    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = await fetch_github_content("https://github.com/test/repo")

        assert result["owner"] == "test"
        assert result["repo"] == "repo"
        assert result["branch"] == "main"
        assert len(result["files"]) == 4  # .gitignore should be filtered out
        assert all(f["path"].endswith((".py", ".md", ".html")) for f in result["files"])


@pytest.mark.asyncio
async def test_fetch_github_file(mock_github_response):
    """Test fetching a single GitHub file"""
    _, _, file_content = mock_github_response

    # Create a mock response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=file_content)

    # Create mock context manager for session.get()
    mock_get_context = AsyncMock()
    mock_get_context.__aenter__ = AsyncMock(return_value=mock_response)
    mock_get_context.__aexit__ = AsyncMock(return_value=None)

    # Create a mock session
    mock_session = AsyncMock()
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = None
    mock_session.get = MagicMock(return_value=mock_get_context)

    # Patch aiohttp.ClientSession to use our mock_session
    with patch("aiohttp.ClientSession", return_value=mock_session):
        content = await fetch_github_file("test", "repo", "src/main.py", {})
        assert content == "Hello World"


@pytest.mark.asyncio
async def test_scrape_recursive_github(mock_github_response, mock_config):
    """Test scraping a GitHub repository"""
    repo_info, tree, file_content = mock_github_response

    # Create mock responses
    mock_repo_response = AsyncMock()
    mock_repo_response.status = 200
    mock_repo_response.json = AsyncMock(return_value=repo_info)

    mock_tree_response = AsyncMock()
    mock_tree_response.status = 200
    mock_tree_response.json = AsyncMock(return_value=tree)

    mock_file_response = AsyncMock()
    mock_file_response.status = 200
    mock_file_response.json = AsyncMock(return_value=file_content)

    # Create mock context managers
    mock_repo_context = AsyncMock()
    mock_repo_context.__aenter__ = AsyncMock(return_value=mock_repo_response)
    mock_repo_context.__aexit__ = AsyncMock(return_value=None)

    mock_tree_context = AsyncMock()
    mock_tree_context.__aenter__ = AsyncMock(return_value=mock_tree_response)
    mock_tree_context.__aexit__ = AsyncMock(return_value=None)

    mock_file_context = AsyncMock()
    mock_file_context.__aenter__ = AsyncMock(return_value=mock_file_response)
    mock_file_context.__aexit__ = AsyncMock(return_value=None)

    # Create a mock session
    mock_session = AsyncMock()
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = None

    # Mock get method to return appropriate context based on URL
    def mock_get(url, headers):
        if "trees" in url:
            return mock_tree_context
        if "contents" in url:
            return mock_file_context
        return mock_repo_context

    mock_session.get = MagicMock(side_effect=mock_get)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        # Mock ChromaDB
        with patch("scraper_chat.scraper.scraper.ChromaHandler") as mock_chroma:
            result = await scrape_recursive(
                "https://github.com/test/repo",
                mock_config["user_agent"],
                mock_config["rate_limit"],
                dump_text=False,
            )

            assert "Successfully processed" in result
            assert (
                mock_chroma.return_value.add_document.call_count == 4
            )  # One for each valid file


def test_get_domain():
    """Test domain extraction from URLs"""
    from scraper_chat.scraper.scraper import get_domain

    test_cases = [
        ("https://example.com/path", "example.com"),
        ("http://subdomain.example.com/page", "subdomain.example.com"),
        ("https://www.test.org/", "www.test.org"),
    ]

    for url, expected in test_cases:
        assert get_domain(url) == expected


@pytest.mark.asyncio
async def test_get_next_proxy(mock_proxies):
    """Test proxy retrieval"""
    import scraper_chat.scraper.scraper

    # Test with empty proxy list
    assert await scraper_chat.scraper.scraper.get_next_proxy() is None

    # Test with proxy list
    mock_proxies.extend(["http://proxy1.com", "http://proxy2.com"])

    # First call should return first proxy
    assert await scraper_chat.scraper.scraper.get_next_proxy() == "http://proxy1.com"


@pytest.mark.asyncio
async def test_switch_to_next_proxy(mock_proxies):
    """Test proxy switching"""
    import scraper_chat.scraper.scraper

    # Prepare proxy list
    mock_proxies.extend(["http://proxy1.com", "http://proxy2.com", "http://proxy3.com"])

    # Initial state
    assert scraper_chat.scraper.scraper.current_proxy_index == 0

    # Switch to next proxy
    await scraper_chat.scraper.scraper.switch_to_next_proxy()
    assert scraper_chat.scraper.scraper.current_proxy_index == 1

    # Switch again
    await scraper_chat.scraper.scraper.switch_to_next_proxy()
    assert scraper_chat.scraper.scraper.current_proxy_index == 2

    # Wrap around to first proxy
    await scraper_chat.scraper.scraper.switch_to_next_proxy()
    assert scraper_chat.scraper.scraper.current_proxy_index == 0


def test_html_to_markdown():
    """Test HTML to markdown conversion"""
    from scraper_chat.scraper.scraper import html_to_markdown

    test_html = """
    <html>
        <body>
            <h1>Test Title</h1>
            <p>Test paragraph</p>
            <div>Another div</div>
        </body>
    </html>
    """

    markdown = html_to_markdown(test_html)

    # Basic checks
    assert "Test Title" in markdown
    assert "Test paragraph" in markdown
    assert "Another div" in markdown


def test_html_to_markdown_config():
    """Test html_to_markdown configuration"""
    import scraper_chat.scraper.scraper

    # Prepare test HTML
    test_html = "<html><body><p>Test Content</p></body></html>"

    # Patch trafilatura config
    with (
        patch("scraper_chat.scraper.scraper.use_config") as mock_use_config,
        patch(
            "scraper_chat.scraper.scraper.text_processor.preprocess_html",
            return_value=test_html,
        ),
    ):
        # Setup mock config
        mock_config = MagicMock()
        mock_use_config.return_value = mock_config

        # Call html_to_markdown
        scraper_chat.scraper.scraper.html_to_markdown(test_html)

        # Verify config setting
        mock_config.set.assert_called_with("DEFAULT", "extraction_timeout", "0")


def test_fetch_github_content_import():
    """Verify fetch_github_content function can be imported"""
    from scraper_chat.scraper.scraper import fetch_github_content

    # Verify it's an async function
    assert asyncio.iscoroutinefunction(fetch_github_content)


def test_fetch_github_file_import():
    """Verify fetch_github_file function can be imported"""
    from scraper_chat.scraper.scraper import fetch_github_file

    # Verify it's an async function
    assert asyncio.iscoroutinefunction(fetch_github_file)


def test_scrape_recursive_import():
    """Verify scrape_recursive function can be imported"""
    from scraper_chat.scraper.scraper import scrape_recursive

    # Verify it's an async function
    assert asyncio.iscoroutinefunction(scrape_recursive)
