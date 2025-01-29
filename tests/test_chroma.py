import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from scraper_chat.database.chroma_handler import ChromaHandler


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton instance before each test"""
    ChromaHandler._instance = None
    ChromaHandler._client = None
    ChromaHandler._collections = {}
    yield


@pytest.fixture
def mock_client():
    client = MagicMock()
    # Mock count method for collections
    collection = MagicMock()
    collection.count.return_value = 5
    client.get_or_create_collection.return_value = collection
    return client


@pytest.fixture
def mock_collection():
    collection = MagicMock()
    collection.query.return_value = {
        "documents": [["Test document"]],
        "metadatas": [[{"url": "http://test.com"}]],  # Fix metadata format
        "distances": [[0.5]],
    }
    collection.count.return_value = 5
    return collection


@pytest.fixture
def chroma_handler(mock_client, mock_collection):
    with patch("chromadb.Client", return_value=mock_client):
        handler = ChromaHandler("test_collection")
        mock_client.get_or_create_collection.return_value = mock_collection
        return handler


def test_singleton_pattern():
    """Test that ChromaHandler follows singleton pattern"""
    with patch("chromadb.Client") as mock_client:
        handler1 = ChromaHandler("test1")
        handler2 = ChromaHandler("test2")
        assert handler1 is handler2
        mock_client.assert_called_once()


def test_get_collection_name():
    """Test URL to collection name conversion"""
    test_cases = [
        ("https://test.com", "test_com"),
        ("http://test-site.com", "test_site_com"),
        ("", "default_collection"),
        ("123.com", "collection_123_com"),
    ]

    for url, expected in test_cases:
        result = ChromaHandler.get_collection_name(url)
        if url.startswith(("http://", "https://")):
            assert result == expected
        elif not url:
            assert result == "default_collection"
        elif url.startswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
            assert result == "collection_" + url.replace(".", "_")


def test_get_collection_name_github():
    """Test GitHub URL to collection name conversion"""
    urls = [
        # Repository URLs
        ("https://github.com/owner/repo", "repo"),
        ("https://github.com/owner/repo/", "repo"),
        # File URLs
        ("https://github.com/owner/repo/blob/main/file.py", "repo"),
        ("https://github.com/owner/repo/blob/main/src/file.py", "repo"),
        # Edge cases
        ("https://github.com/owner/repo-name", "repo-name"),
        ("https://github.com/owner/repo.js", "repo.js"),
    ]

    for url, expected in urls:
        assert ChromaHandler.get_collection_name(url) == expected


def test_add_document(chroma_handler, mock_collection):
    """Test adding a document"""
    # Reset collection cache
    ChromaHandler._collections = {}

    with patch.object(chroma_handler, "get_collection", return_value=mock_collection):
        chroma_handler.add_document("Test content", "http://test.com")
        mock_collection.upsert.assert_called_once()

        # Test empty content
        chroma_handler.add_document("", "http://test.com")
        # Should still be called only once (empty content skipped)
        mock_collection.upsert.assert_called_once()


def test_add_document_github(chroma_handler, mock_collection):
    """Test adding GitHub file content without chunking"""
    # Test GitHub file URL
    github_url = "https://github.com/owner/repo/blob/main/file.py"
    content = "def hello():\n    return 'world'"

    chroma_handler.add_document(content, github_url)

    # Verify the document was added without chunking
    mock_collection.upsert.assert_called_once()
    call_args = mock_collection.upsert.call_args[1]

    assert len(call_args["documents"]) == 1  # No chunks
    assert call_args["documents"][0] == content
    assert call_args["metadatas"][0]["url"] == github_url
    assert "chunk_index" not in call_args["metadatas"][0]  # No chunking metadata


def test_add_document_regular_vs_github(chroma_handler, mock_collection):
    """Test different handling of regular vs GitHub content"""
    # Regular web content should be chunked
    web_url = "https://example.com/docs"
    web_content = "A" * 2000  # Long content that would normally be chunked

    chroma_handler.add_document(web_content, web_url)
    web_call = mock_collection.upsert.call_args[1]

    # GitHub content should not be chunked
    mock_collection.reset_mock()
    github_url = "https://github.com/owner/repo/blob/main/file.py"
    github_content = "B" * 2000

    chroma_handler.add_document(github_content, github_url)
    github_call = mock_collection.upsert.call_args[1]

    # Web content should have multiple chunks
    assert len(web_call["documents"]) > 1
    assert all("chunk_index" in meta for meta in web_call["metadatas"])

    # GitHub content should be a single document
    assert len(github_call["documents"]) == 1
    assert "chunk_index" not in github_call["metadatas"][0]


def test_query(chroma_handler, mock_collection):
    """Test querying documents"""
    with patch.object(chroma_handler, "get_collection", return_value=mock_collection):
        results = chroma_handler.query("test_collection", "test query")
        assert len(results) == 1
        assert results[0]["text"] == "Test document"
        assert results[0]["url"] == "http://test.com"
        assert results[0]["distance"] == 0.5


def test_delete_collection():
    """Test deleting a collection"""
    mock_client = MagicMock()

    with patch("chromadb.Client", return_value=mock_client) as mock_client_class:
        result = ChromaHandler.delete_collection("test_collection")
        assert result is True
        mock_client.delete_collection.assert_called_once_with("test_collection")


def test_delete_collection_error():
    """Test error handling in delete_collection"""
    mock_client = MagicMock()
    mock_client.delete_collection.side_effect = Exception("Test error")

    with patch("chromadb.Client", return_value=mock_client):
        result = ChromaHandler.delete_collection("test_collection")
        assert result is False


def test_get_available_collections():
    """Test getting available collections"""
    mock_client = MagicMock()
    mock_client.list_collections.return_value = ["collection1", "collection2"]

    with patch("chromadb.Client", return_value=mock_client):
        collections = ChromaHandler.get_available_collections()
        assert collections == ["collection1", "collection2"]
        mock_client.list_collections.assert_called_once()
