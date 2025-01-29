import asyncio
import pytest
from unittest.mock import patch
import os
import shutil
from pathlib import Path
import json
import logging
import sys
import time
from urllib.parse import urlparse
import aiohttp
import trafilatura
from bs4 import BeautifulSoup
import tempfile
from chromadb.config import Settings
from scraper_chat.embeddings.embeddings import Reranker
from scraper_chat.chat.chat_interface import ChatInterface
from scraper_chat.database.chroma_handler import ChromaHandler

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logger = logging.getLogger(__name__)


def extract_text_from_html(html: str) -> str:
    """Extract text from HTML using trafilatura"""
    text = trafilatura.extract(html, include_links=True, include_formatting=True)
    if not text:
        # Fallback to basic extraction if trafilatura fails
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
    return text or ""


@pytest.fixture
def chroma_settings(test_db):
    """Create ChromaDB settings"""
    return Settings(
        allow_reset=True,
        is_persistent=True,
        persist_directory=test_db,
        anonymized_telemetry=False,
    )


@pytest.fixture
def test_db():
    """Create a test ChromaDB directory"""
    db_path = os.path.join(tempfile.gettempdir(), f"chroma_test_{int(time.time())}")
    yield db_path


@pytest.fixture(autouse=True)
def clean_chroma_db(test_db):
    """Clean ChromaDB directory before and after each test"""
    # Before test
    ChromaHandler._instance = None
    ChromaHandler._client = None
    ChromaHandler._collections = {}
    ChromaHandler.configure(test_db)  # Configure to use test path

    if os.path.exists(test_db):
        try:
            shutil.rmtree(test_db)
        except (PermissionError, OSError) as e:
            logger.warning(f"Could not clean up {test_db} before test: {e}")

    yield

    # After test
    if ChromaHandler._client:
        try:
            ChromaHandler._client.reset()
        except Exception as e:
            logger.warning(f"Error resetting ChromaDB client: {e}")

    # Reset ChromaDB handler to default state
    ChromaHandler._client = None
    ChromaHandler._instance = None
    ChromaHandler._collections = {}
    ChromaHandler.configure(None)  # Reset to default path

    # Give ChromaDB time to release files
    time.sleep(0.5)

    # Clean up temporary directory
    retries = 3
    for i in range(retries):
        try:
            if os.path.exists(test_db):
                shutil.rmtree(test_db)
            break
        except Exception as e:
            if i == retries - 1:
                logger.warning(f"Could not clean up {test_db}: {e}")
            else:
                time.sleep(0.5)


@pytest.fixture
async def test_server():
    """Create and run a test HTTP server"""
    from test_server import TestServer

    server = TestServer()
    url = await server.start()
    yield url
    await server.stop()


@pytest.fixture
def test_config():
    """Create a test config file"""
    config = {
        "llm": {"model": "gpt-3.5-turbo", "temperature": 0.7, "max_tokens": 500},
        "scraping": {"user_agent": "TestBot/1.0", "rate_limit": 1, "max_retries": 3},
    }
    config_path = "test_real_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    yield config_path
    if os.path.exists(config_path):
        os.remove(config_path)


@pytest.fixture(autouse=True)
async def cleanup_tasks():
    """Clean up any pending tasks after each test"""
    yield
    # Cancel all tasks
    for task in asyncio.all_tasks():
        if task != asyncio.current_task():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


def get_safe_collection_name(url: str) -> str:
    """Convert URL to a valid ChromaDB collection name"""
    parsed = urlparse(url)
    # Use hostname without port as base
    collection_name = parsed.hostname.replace(".", "_")
    # Add test-specific prefix and ensure no special characters
    collection_name = f"test_{collection_name}_{int(time.time())}"
    collection_name = collection_name.replace("-", "_")
    return collection_name


@pytest.fixture
def reranker():
    """Create a reranker instance for testing"""
    # Reset singleton state
    Reranker._instance = None
    Reranker._rerank_model = None
    return Reranker()


@pytest.mark.asyncio
async def test_data_flow(test_server, test_db, test_config, reranker):
    """Test data passing between components"""
    with patch("scraper_chat.config.CONFIG_FILE", test_config):
        try:
            # 1. Scrape specific page with timeout
            page_url = f"{test_server}/page1"

            async def mock_scrape(url, user_agent, rate_limit=1, use_db=True):
                logger.debug(f"Mock scraping {url}")
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        logger.debug(f"Response status: {response.status}")
                        html = await response.text()
                        logger.debug(f"Response content: {html[:200]}...")
                        response.raise_for_status()

                # Process the page directly
                text = extract_text_from_html(html)
                logger.debug(f"Extracted text: {text[:200]}...")

                if text and use_db:
                    # Get collection name from URL
                    collection_name = ChromaHandler.get_collection_name(url)
                    logger.debug(f"Using collection: {collection_name}")

                    # Add document
                    db = ChromaHandler(collection_name)
                    db.add_document(text, url)
                    logger.debug(f"Added document to collection {collection_name}")

            async with asyncio.timeout(10):  # Increase timeout for scraping
                await mock_scrape(page_url, "TestBot/1.0")

            # Wait a moment for ChromaDB to process
            await asyncio.sleep(0.5)

            # 2. Verify exact content in ChromaDB
            collection_name = ChromaHandler.get_collection_name(page_url)
            db = ChromaHandler(collection_name)

            # Debug output
            logger.debug(f"Using collection: {collection_name}")

            # Query for results
            results = db.query(collection_name, "Feature 1", n_results=1)
            logger.debug(f"Query results: {results}")

            assert len(results) == 1
            result = results[0]

            # Check data integrity
            assert "Feature 1" in result["text"]
            assert "core component" in result["text"]
            assert "example_function" in result["text"]
            assert page_url in result["url"]

            # 3. Verify chat uses correct context with timeout
            chat = ChatInterface([collection_name])

            # Disable rate limiting for test
            from scraper_chat.core.llm_config import LLMConfig

            LLMConfig.configure_rate_limit(0)  # 0 RPM disables rate limiting

            responses = []
            excerpts_seen = []
            async with asyncio.timeout(5):  # 5 second timeout
                async for response, excerpts in chat.get_response("What is Feature 1?"):
                    responses.append(response)
                    if excerpts:
                        excerpts_seen.extend(excerpts)

            assert len(responses) > 0
            assert len(excerpts_seen) > 0

            # Verify context matches original document
            excerpt = excerpts_seen[0]
            assert "Feature 1" in excerpt["text"]
            assert "core component" in excerpt["text"]
            assert page_url in excerpt["url"]

            # Test querying with reranking
            query = "test query"
            results = db.query(
                collection_name, query
            )  # Don't specify n_results to get all available

            # Verify we got all available chunks
            assert len(results) > 0
            total_chunks = results[0]["metadata"]["total_chunks"]
            assert len(results) == total_chunks
            assert all("text" in doc and "url" in doc for doc in results)

            # Test reranking with all available documents
            documents = [doc["text"] for doc in results]
            reranked_indices = reranker.rerank(query, documents, top_k=len(documents))

            # Verify reranking results
            assert len(reranked_indices) == len(documents)
            assert all(isinstance(idx, int) for idx in reranked_indices)
            assert all(0 <= idx < len(documents) for idx in reranked_indices)
            assert len(set(reranked_indices)) == len(
                reranked_indices
            )  # Check all indices are unique

            # Verify reranked documents are properly ordered
            reranked_docs = [documents[i] for i in reranked_indices]
            assert len(reranked_docs) == len(documents)
            assert all(isinstance(doc, str) for doc in reranked_docs)
        finally:
            # Clean up any pending tasks
            for task in asyncio.all_tasks():
                if task != asyncio.current_task():
                    task.cancel()
