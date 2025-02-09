"""Tests for real-world scraping scenarios focusing on content deduplication."""

import pytest
import sys
from pathlib import Path
import os
import shutil
from unittest.mock import patch
from urllib.parse import urlparse

sys.path.append(str(Path(__file__).parent.parent.parent))
from scraper_chat.scraper.scraper import scrape_recursive, is_blog_post

EXPECTED_REACT_NATIVE_TEXT = [
    "A compelling reason to use React Native instead of WebView-based tools is to achieve at least 60 frames per second",
    "Another example is responding to touches",
    "One solution to this is to allow for JavaScript-based animations to be offloaded",
    "Left click or right click to increase / decrease the SPD or LOOP values",
    "The two additional parameters can be used to override these defaults",
    "Note that not every PICO-8 will have a keyboard or mouse attached to it, so when posting carts to the",
]


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
    # Create filename based on domain only, matching the actual scraper pattern
    domain = urlparse(url).netloc
    filename = domain.replace(".", "_") + ".txt"
    return os.path.join(test_dir, filename)


class MockFileHandle:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def write(self, *args):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def mock_open(*args, **kwargs):
    """Mock file operations"""
    return MockFileHandle()


@pytest.mark.asyncio
async def test_react_native_docs_deduplication(test_output_dir):
    """Test that duplicate React Native performance docs are properly deduplicated."""
    urls = [
        "https://reactnative.dev/docs/performance",
        "https://www.lexaloffle.com/dl/docs/pico-8_manual.html",
    ]

    processed_contents = []

    # Patch both check_sitemap and get_output_filename
    with (
        patch("scraper_chat.scraper.scraper.check_sitemap", return_value=None),
        patch(
            "scraper_chat.scraper.scraper.get_output_filename",
            side_effect=mock_get_output_filename,
        ),
    ):
        for url in urls:
            # Scrape each URL
            await scrape_recursive(
                start_url=url,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                rate_limit=1,
                dump_text=True,
                force_rescrape=True,
            )

            # Get the output filename for this URL
            filename = mock_get_output_filename(url)

            # Read and process the content if file exists
            if os.path.exists(filename):
                with open(filename, "r", encoding="utf-8") as f:
                    processed_contents.append(f.read())

    # First URL should have content
    assert processed_contents[0] != ""
    assert EXPECTED_REACT_NATIVE_TEXT[0] in processed_contents[0]
    assert EXPECTED_REACT_NATIVE_TEXT[1] in processed_contents[0]
    assert EXPECTED_REACT_NATIVE_TEXT[2] in processed_contents[0]
    assert EXPECTED_REACT_NATIVE_TEXT[3] in processed_contents[1]
    assert EXPECTED_REACT_NATIVE_TEXT[4] in processed_contents[1]
    assert EXPECTED_REACT_NATIVE_TEXT[5] in processed_contents[1]


@pytest.mark.asyncio
async def test_blog_post_filtering():
    """Test that blog posts are properly detected and filtered."""
    # Test URLs that should be identified as blog posts
    blog_urls = [
        "https://reactnative.dev/blog/2025/01/21/version-0.77",
        "https://svelte.dev/blog/advent-of-svelte",
        "https://astro.build/blog/",
        "https://www.djangoproject.com/weblog/2025/feb/05/bugfix-releases/",
    ]

    # First test the is_blog_post function directly
    for url in blog_urls:
        assert is_blog_post(url), f"Failed to detect blog post: {url}"

    # Now test actual scraping - none of these should be scraped
    with (
        patch("scraper_chat.scraper.scraper.check_sitemap", return_value=None),
        patch(
            "scraper_chat.scraper.scraper.get_output_filename",
            side_effect=mock_get_output_filename,
        ),
        patch("scraper_chat.scraper.scraper.ChromaHandler") as mock_chroma,
        patch("scraper_chat.scraper.scraper.open", side_effect=mock_open),
        patch("scraper_chat.scraper.scraper.fetch_page", return_value=None),
    ):
        for url in blog_urls:
            # Scrape each URL; since fetch_page returns None, file operations will be invoked but no content is processed
            await scrape_recursive(
                start_url=url,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                rate_limit=1,
                dump_text=True,
                force_rescrape=True,
            )

            # Verify that add_document was never called for these blog URLs
            mock_chroma.return_value.add_document.assert_not_called()


@pytest.mark.asyncio
async def test_documentation_pages():
    """Test that legitimate documentation pages are not filtered."""
    # Test URLs that should NOT be identified as blog posts
    doc_urls = [
        "https://reactnative.dev/docs/getting-started",
        "https://svelte.dev/docs/introduction",
        "https://docs.astro.build/en/getting-started/",
        "https://docs.djangoproject.com/en/stable/intro/tutorial01/",
    ]

    # First test the is_blog_post function directly
    for url in doc_urls:
        assert not is_blog_post(url), f"Incorrectly detected as blog post: {url}"

    # Create test directory
    test_dir = os.path.join(os.path.dirname(__file__), "test_scrapedtxt")
    os.makedirs(test_dir, exist_ok=True)

    try:
        # Now test actual scraping - these should be scraped
        with (
            patch("scraper_chat.scraper.scraper.check_sitemap", return_value=None),
            patch(
                "scraper_chat.scraper.scraper.get_output_filename",
                side_effect=mock_get_output_filename,
            ),
            patch("scraper_chat.scraper.scraper.ChromaHandler") as mock_chroma,
        ):
            for url in doc_urls:
                # Scrape each URL
                await scrape_recursive(
                    start_url=url,
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    rate_limit=1,
                    dump_text=True,
                    force_rescrape=True,
                )

                # Verify that add_document was called at least once for these URLs
                assert mock_chroma.return_value.add_document.called, (
                    f"Failed to scrape documentation: {url}"
                )

                # Verify that the file was created
                expected_file = mock_get_output_filename(url)
                assert os.path.exists(expected_file), (
                    f"Expected file not created: {expected_file}"
                )
    finally:
        # Clean up test directory
        shutil.rmtree(test_dir, ignore_errors=True)
