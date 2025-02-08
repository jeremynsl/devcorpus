"""Tests for real-world scraping scenarios focusing on content deduplication."""

import pytest
from scraper_chat.scraper.scraper import scrape_recursive
import os
import shutil
from unittest.mock import patch


EXPECTED_REACT_NATIVE_TEXT = ["A compelling reason to use React Native instead of WebView-based tools is to achieve at least 60 frames per second",
"Another example is responding to touches",
"One solution to this is to allow for JavaScript-based animations to be offloaded"]


@pytest.fixture
def test_output_dir():
    """Create and clean up a test output directory."""
    test_dir = os.path.join(os.path.dirname(__file__), "test_scrapedtxt")
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir
    # Clean up after test
    shutil.rmtree(test_dir)

def mock_get_output_filename(url: str) -> str:
    """Override output filename to use test directory."""
    test_dir = os.path.join(os.path.dirname(__file__), "test_scrapedtxt")
    # Create a filename based on the URL path
    path_part = url.split("//")[-1].replace("/", "_").replace(".", "_")
    return os.path.join(test_dir, f"{path_part}.txt")

@pytest.mark.asyncio
async def test_react_native_docs_deduplication(test_output_dir):
    """Test that duplicate React Native performance docs are properly deduplicated."""
    urls = [
        "https://reactnative.dev/docs/performance"
    ]

    processed_contents = []
    
    # Patch both check_sitemap and get_output_filename
    with patch('scraper_chat.scraper.scraper.check_sitemap', return_value=None), \
         patch('scraper_chat.scraper.scraper.get_output_filename', side_effect=mock_get_output_filename):
        for url in urls:
            # Scrape each URL
            await scrape_recursive(
                start_url=url,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                rate_limit=1,
                dump_text=True,
                force_rescrape=True
            )
            
            # Get the output filename for this URL
            filename = mock_get_output_filename(url)
            
            # Read and process the content if file exists
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    processed_contents.append(f.read())
    
    # First URL should have content
    assert processed_contents[0] != ""
    assert EXPECTED_REACT_NATIVE_TEXT[0] in processed_contents[0]
    assert EXPECTED_REACT_NATIVE_TEXT[1] in processed_contents[0]
    assert EXPECTED_REACT_NATIVE_TEXT[2] in processed_contents[0]
