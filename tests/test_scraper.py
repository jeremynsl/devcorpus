import pytest
from unittest.mock import patch, AsyncMock, mock_open, MagicMock
import sys
from pathlib import Path
import asyncio

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from main import (
    get_output_filename, remove_anchor, fetch_page, extract_links,
     scrape_recursive
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
    import main
    main.proxies_list = []
    return main.proxies_list

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
    mock_response = AsyncMock()
    mock_response.text = "Test content"
    mock_response.raise_for_status = AsyncMock()  # Add this method
    
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value.get.return_value = mock_response
    
    with patch("httpx.AsyncClient", return_value=mock_client):
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
    
    with patch('main.extract_metadata', return_value=mock_metadata):
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
         patch("main.fetch_page", side_effect=mock_fetch), \
         patch("main.tqdm"), \
         patch("main.watch_for_input", AsyncMock()):
        
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
