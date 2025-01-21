import pytest
from unittest.mock import patch, AsyncMock, mock_open, MagicMock
import sys
from pathlib import Path
import asyncio

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from scraper import (
    get_output_filename, remove_anchor, fetch_page, extract_links,
     scrape_recursive, fetch_github_content, fetch_github_file
)

@pytest.fixture
def mock_config():
    """Create a mock config"""
    return {
        "proxies": [],
        "rate_limit": 2,
        "user_agent": "TestBot/1.0"
    }

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
    import scraper
    scraper.proxies_list = []
    return scraper.proxies_list

@pytest.fixture
def mock_github_response():
    """Create mock GitHub API responses"""
    repo_info = {
        "default_branch": "main"
    }
    
    tree = {
        "tree": [
            {"path": "src/main.py", "type": "blob"},
            {"path": "README.md", "type": "blob"},
            {"path": "tests/test_main.py", "type": "blob"},
            {"path": "docs/index.html", "type": "blob"},
            {"path": ".gitignore", "type": "blob"}  # Should be filtered out
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
        ("https://test.org/path", os.path.join(scrapedtxt_dir, "test_org.txt"))
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
    mock_response.text.return_value = "Test content"
    mock_response.raise_for_status = AsyncMock()

    # Create a mock context manager for session.get()
    mock_get_context = AsyncMock()
    mock_get_context.__aenter__ = AsyncMock(return_value=mock_response)
    mock_get_context.__aexit__ = AsyncMock(return_value=None)

    # Create a mock session
    mock_session = AsyncMock()
    # Setup the session so that its async context manager returns itself
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = None

    # Use a synchronous MagicMock for the get method
    mock_session.get = MagicMock(return_value=mock_get_context)

    # Patch aiohttp.ClientSession to use our mock_session
    with patch("aiohttp.ClientSession", return_value=mock_session):
        content = await fetch_page("http://test.com", "TestBot/1.0")

    assert content == "Test content"

def test_extract_links(mock_html):
    """Test link extraction"""
    # Mock trafilatura metadata
    mock_metadata = MagicMock()
    mock_metadata.as_dict.return_value = {
        'links': [
            'http://example.com/start/page1',
            'http://example.com/start/page2',
            'http://example.com/start/page3'
        ]
    }
    
    with patch('scraper.extract_metadata', return_value=mock_metadata):
        links = extract_links(
            mock_html,
            "http://example.com/start",
            "example.com",
            "/start/"
        )
        assert len(links) == 3
        assert all(link.startswith("http://example.com/start/") for link in links)

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
    with patch("builtins.open", m), \
         patch("scraper.fetch_page", side_effect=mock_fetch), \
         patch("scraper.tqdm"), \
         patch("scraper.watch_for_input", AsyncMock()):
        
        async def run_with_timeout():
            try:
                async with asyncio.timeout(2):
                    await scrape_recursive(
                        "http://test.com",
                        "TestBot/1.0",
                        rate_limit=2,
                        use_db=False
                    )
            except asyncio.TimeoutError:
                # Get all tasks that are still running
                tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
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
        with patch('scraper.ChromaHandler') as mock_chroma:
            result = await scrape_recursive(
                "https://github.com/test/repo",
                mock_config["user_agent"],
                mock_config["rate_limit"],
                use_db=True
            )
            
            assert "Successfully processed" in result
            assert mock_chroma.return_value.add_document.call_count == 4  # One for each valid file
