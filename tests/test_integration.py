import pytest
import os
import shutil
import asyncio
from pathlib import Path
import json
from unittest.mock import patch, MagicMock, mock_open
import sys

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from scraper_chat.scraper.scraper import scrape_recursive
from scraper_chat.chat.chat_interface import ChatInterface
from scraper_chat.database.chroma_handler import ChromaHandler


@pytest.fixture(autouse=True)
def setup_test_env(test_db):
    """Set up test environment"""
    # Configure ChromaDB to use test path
    ChromaHandler.configure(test_db)
    yield
    # Reset to default after test
    ChromaHandler.configure(None)


@pytest.fixture
def setup_chroma():
    """Reset ChromaHandler singleton between tests"""
    ChromaHandler.reset()
    yield
    ChromaHandler.reset()


@pytest.fixture
def test_config():
    """Create a test config file"""
    config = {
        "proxies": [],
        "rate_limit": 2,
        "user_agent": "TestBot/1.0",
        "chat": {
            "message_history_size": 5,
            "system_prompt": "Test system prompt",
            "models": {"available": ["test-model"], "default": "test-model"},
        },
    }
    config_path = "test_scraper_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    yield config_path
    if os.path.exists(config_path):
        os.remove(config_path)


@pytest.fixture
def test_db():
    """Create a test ChromaDB directory"""
    db_path = "test_docs_database.db"
    # Before test
    if os.path.exists(db_path):
        try:
            shutil.rmtree(db_path)
        except (PermissionError, OSError):
            pass  # Directory might be locked, will be cleaned up later
    yield db_path
    # After test
    try:
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
    except (PermissionError, OSError):
        pass  # Directory might be locked, will be cleaned up later


@pytest.fixture
def mock_html_content():
    """Create mock HTML content"""
    return {
        "page1": """
            <html><body>
                <h1>Test Documentation</h1>
                <p>This is a test documentation page. It contains important documentation content here.</p>
                <p>Documentation includes several features:</p>
                <ul>
                    <li>Feature 1: Important documentation details</li>
                    <li>Feature 2: More documentation content</li>
                </ul>
                <a href="/page2">Next Page</a>
            </body></html>
        """,
        "page2": """
            <html><body>
                <h2>More Documentation</h2>
                <p>Additional documentation content on page 2.</p>
                <a href="/page1">Back to Page 1</a>
            </body></html>
        """,
    }


@pytest.fixture
def mock_config():
    return {
        "chat": {
            "system_prompt": "Test system prompt",
            "models": {"available": ["test-model"], "default": "test-model"},
            "message_history_size": 10,
            "rag_prompt": "Context:\n{context}\n\nQuestion: {query}",
        }
    }


@pytest.fixture
def mock_collection():
    collection = MagicMock()
    # Mock query to return documentation content
    collection.query.return_value = {
        "documents": [
            [
                "This is a test documentation page. It contains important documentation content here."
            ]
        ],
        "metadatas": [[{"url": "http://test.com/page1"}]],
        "distances": [[0.5]],
    }
    collection.count.return_value = 5
    collection.upsert = MagicMock()
    collection.get = MagicMock(return_value={"ids": [], "metadatas": []})
    return collection


@pytest.fixture
def chroma_handler(mock_collection):
    handler = ChromaHandler("test_collection")
    # Mock has_matching_content to always return False
    handler.has_matching_content = MagicMock(return_value=False)
    # Mock the chunking manager to return actual content
    handler._chunking_manager.chunk_text = MagicMock(
        return_value=[
            "This is a test documentation page. It contains important documentation content here."
        ]
    )
    mock_collection.get_or_create_collection.return_value = mock_collection
    return handler


@pytest.mark.asyncio
async def test_full_scrape_and_chat_workflow(
    test_config, test_db, mock_html_content, mock_collection
):
    """Test the full workflow: scraping, storing in DB, and chatting"""
    # Configure ChromaDB to use test path
    ChromaHandler.configure(test_db)

    # Mock ChromaDB client and collection
    with patch.object(ChromaHandler, "_client") as mock_client:
        mock_client.get_or_create_collection.return_value = mock_collection

        # Mock HTTP responses
        async def mock_fetch(*args, **kwargs):
            url = args[0]
            if "page1" in url:
                return mock_html_content["page1"]
            elif "page2" in url:
                return mock_html_content["page2"]
            return ""

        # Mock LLM responses
        async def mock_llm_response(*args, **kwargs):
            async def response_gen():
                yield MagicMock(
                    choices=[
                        MagicMock(
                            delta=MagicMock(content="Based on the documentation, ")
                        )
                    ]
                )

            return response_gen()

        with (
            patch("scraper_chat.scraper.scraper.fetch_page", side_effect=mock_fetch),
            patch(
                "scraper_chat.core.llm_config.LLMConfig.get_response",
                side_effect=mock_llm_response,
            ),
            patch("scraper_chat.config.CONFIG_FILE", test_config),
            # Mock the chunking manager to return actual content
            patch.object(ChromaHandler, "_chunking_manager") as mock_chunking_manager,
        ):
            mock_chunking_manager.chunk_text.return_value = [
                "This is a test documentation page. It contains important documentation content here."
            ]

            # 1. Scrape documentation
            await scrape_recursive(
                start_url="http://test.com/page1",
                user_agent="TestBot/1.0",
                rate_limit=2,
                dump_text=True,
                force_rescrape=False,
            )

            # 2. Verify ChromaDB has content
            collection_name = ChromaHandler.get_collection_name("http://test.com/page1")
            db = ChromaHandler(collection_name)
            results = db.query(collection_name, "documentation", n_results=5)
            assert len(results) > 0
            assert any("documentation" in result["text"].lower() for result in results)

            # 3. Test chat interaction
            chat = ChatInterface([collection_name])
            responses = []
            async for response, excerpts in chat.get_response(
                "What is in the documentation?"
            ):
                responses.append(response)
                assert len(excerpts) > 0  # Should have relevant excerpts

            assert len(responses) > 0
            assert "documentation" in " ".join(responses).lower()


@pytest.mark.asyncio
async def test_multi_collection_chat(test_config, test_db, mock_html_content):
    """Test chatting with multiple collections"""

    # Mock LLM responses for different queries
    class AsyncLLMResponse:
        def __init__(self):
            self.chunks = [
                MagicMock(
                    choices=[
                        MagicMock(
                            delta=MagicMock(
                                content="Combined information from both sources: "
                            )
                        )
                    ]
                )
            ]
            self.index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index >= len(self.chunks):
                raise StopAsyncIteration
            chunk = self.chunks[self.index]
            self.index += 1
            return chunk

    async def mock_llm_response(*args, **kwargs):
        return AsyncLLMResponse()

    # Mock ChromaDB query results
    mock_results1 = {
        "documents": [["Documentation from source 1"]],
        "metadatas": [[{"url": "http://test1.com"}]],
        "distances": [[0.1]],
    }

    mock_results2 = {
        "documents": [["Documentation from source 2"]],
        "metadatas": [[{"url": "http://test2.com"}]],
        "distances": [[0.2]],
    }

    with patch(
        "scraper_chat.core.llm_config.LLMConfig.get_response",
        side_effect=mock_llm_response,
    ):
        # Create mock collections
        mock_collection1 = MagicMock()
        mock_collection1.query.return_value = mock_results1
        mock_collection1.count.return_value = 1

        mock_collection2 = MagicMock()
        mock_collection2.query.return_value = mock_results2
        mock_collection2.count.return_value = 1

        # Mock ChromaHandler to return our mock collections
        with patch.object(ChromaHandler, "get_collection") as mock_get_collection:

            def get_collection_side_effect(name):
                if name == "test_collection1":
                    return mock_collection1
                return mock_collection2

            mock_get_collection.side_effect = get_collection_side_effect

            # Test chat with both collections
            chat = ChatInterface(["test_collection1", "test_collection2"])
            responses = []
            excerpts_seen = []
            async for response, excerpts in chat.get_response(
                "Show me all documentation"
            ):
                responses.append(response)
                if excerpts:
                    excerpts_seen.extend(excerpts)

            assert len(responses) > 0
            assert len(excerpts_seen) > 0
            combined_response = " ".join(responses).lower()
            assert "combined information" in combined_response

            # Verify we got content from both collections
            urls = {excerpt.get("url") for excerpt in excerpts_seen}
            assert "http://test1.com" in urls
            assert "http://test2.com" in urls


@pytest.mark.asyncio
async def test_error_recovery(test_config, test_db, mock_config):
    """Test system recovery from various errors"""

    # Test scraping with failing URLs
    async def mock_failing_fetch(*args, **kwargs):
        raise Exception("Network error")

    # Mock tqdm to avoid progress bar issues
    mock_tqdm = MagicMock()
    mock_tqdm.update = lambda x: None
    mock_tqdm.close = lambda: None
    mock_tqdm.n = 0
    mock_tqdm.format_dict = {"rate": 0}

    # Mock file operations
    mock_file = mock_open(read_data=json.dumps(mock_config))

    # Create a timeout for the scraping
    async def run_with_timeout():
        try:
            async with asyncio.timeout(2):  # 2 second timeout
                await scrape_recursive(
                    start_url="http://test.com/error",
                    user_agent="TestBot/1.0",
                    rate_limit=1,  # Use single worker for simpler testing
                    dump_text=False,
                    force_rescrape=False,
                )
        except asyncio.TimeoutError:
            # Expected timeout since failing fetches never complete
            pass
        except Exception as e:
            # Other exceptions should still be raised
            raise e

    with (
        patch(
            "scraper_chat.scraper.scraper.fetch_page", side_effect=mock_failing_fetch
        ),
        patch("scraper_chat.scraper.scraper.tqdm", return_value=mock_tqdm),
        patch("builtins.open", mock_file),
        patch("scraper_chat.config.CONFIG_FILE", test_config),
    ):
        # Run scraping with timeout
        await run_with_timeout()

        # ChromaDB should still be accessible
        db = ChromaHandler("test_collection")
        assert db is not None

        # Chat interface should handle empty results gracefully
        chat = ChatInterface(["test_collection"])

        responses = []
        async for response, excerpts in chat.get_response("test query"):
            responses.append(response)
            assert len(excerpts) == 0

        assert any("no relevant information" in r.lower() for r in responses)
