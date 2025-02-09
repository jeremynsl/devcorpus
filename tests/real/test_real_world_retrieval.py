"""Tests for real-world content retrieval from ChromaDB."""

import pytest
import sys
from pathlib import Path

from unittest.mock import patch
import os
import shutil

sys.path.append(str(Path(__file__).parent.parent.parent))
from scraper_chat.scraper.scraper import scrape_recursive
from scraper_chat.database.chroma_handler import ChromaHandler

# URLs to test retrieval from
URLS = ["https://www.lexaloffle.com/dl/docs/pico-8_manual.html"]

EXPECTED_METATABLE_TEXT = "Metatables can be used to define the behaviour of objects under particular operations"
EXPECTED_TEXT_2 = "Draw a textured line from (X0,Y0) to (X1,Y1)"
EXPECTED_TEXT_3 = "Base RAM (64k)"
EXPECTED_TEXT_4 = "This is useful for instruments that change more slowly over time"


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
    # Create filename based on domain only
    domain = url.split("//")[-1].split("/")[0]
    filename = domain.replace(".", "_") + ".txt"
    return os.path.join(test_dir, filename)


@pytest.mark.asyncio
async def test_metatable_retrieval(test_output_dir):
    """Test retrieving metatable information from PICO-8 documentation."""

    # First scrape the content into ChromaDB
    with patch(
        "scraper_chat.scraper.scraper.get_output_filename",
        side_effect=mock_get_output_filename,
    ):
        for url in URLS:
            await scrape_recursive(
                start_url=url,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                rate_limit=1,
                dump_text=True,
                force_rescrape=True,
            )

    # Now query ChromaDB for metatable information
    collection_name = ChromaHandler.get_collection_name(URLS[0])
    db_handler = ChromaHandler(collection_name)
    results, avg_score = db_handler.query(
        collection_name, "How to create a metatable", n_results=5
    )

    # Debug print results
    print("\nQuery Results for metatables:")
    print(f"Average Score: {avg_score}")
    for i, result in enumerate(results):
        print(f"\nResult {i + 1}:")
        print(f"Text: {result['text']}")
        print(f"URL: {result['url']}")
        print(f"Distance: {result['distance']}")
        print(f"Rerank Score: {result.get('rerank_score', 'N/A')}")

    # Check if expected text about metatables is in the results
    print(
        f"\nExpected text ({len(EXPECTED_METATABLE_TEXT)} chars): {repr(EXPECTED_METATABLE_TEXT)}"
    )
    found = False
    for i, result in enumerate(results):
        actual_text = result["text"]
        print(f"\nResult {i + 1} text ({len(actual_text)} chars): {repr(actual_text)}")
        if EXPECTED_METATABLE_TEXT in actual_text:
            print(f"Found in result {i + 1}!")
            found = True
            break
    if not found:
        print("Text not found in any result!")
    assert found, f"Expected text about metatables not found in query results"

    # Second query
    print("\n=== Second Query ===")
    results, avg_score = db_handler.query(
        collection_name, "How to draw a textured line", n_results=5
    )
    print("\nQuery Results:")
    print(f"Average Score: {avg_score}")
    for i, result in enumerate(results):
        print(f"\nResult {i + 1}:")
        print(f"Text: {result['text']}")
        print(f"URL: {result['url']}")
        print(f"Distance: {result['distance']}")
        print(f"Rerank Score: {result.get('rerank_score', 'N/A')}")

    found = any(EXPECTED_TEXT_2 in result["text"] for result in results)
    assert found, f"Expected text not found in second query results"

    # Third query
    print("\n=== Third Query ===")
    results, avg_score = db_handler.query(
        collection_name, "What types of memory does PICO-8 use?", n_results=5
    )
    print("\nQuery Results:")
    print(f"Average Score: {avg_score}")
    for i, result in enumerate(results):
        print(f"\nResult {i + 1}:")
        print(f"Text: {result['text']}")
        print(f"URL: {result['url']}")
        print(f"Distance: {result['distance']}")
        print(f"Rerank Score: {result.get('rerank_score', 'N/A')}")

    found = any(EXPECTED_TEXT_3 in result["text"] for result in results)
    assert found, f"Expected text not found in third query results"

    # Fourth query
    print("\n=== Fourth Query ===")
    results, avg_score = db_handler.query(
        collection_name, "How are SFX instruments used?", n_results=5
    )
    print("\nQuery Results:")
    print(f"Average Score: {avg_score}")
    for i, result in enumerate(results):
        print(f"\nResult {i + 1}:")
        print(f"Text: {result['text']}")
        print(f"URL: {result['url']}")
        print(f"Distance: {result['distance']}")
        print(f"Rerank Score: {result.get('rerank_score', 'N/A')}")

    found = any(EXPECTED_TEXT_4 in result["text"] for result in results)
    assert found, f"Expected text not found in fourth query results"
