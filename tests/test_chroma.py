import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
from hashlib import blake2b
from chromadb.config import Settings

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
        "metadatas": [[{"url": "http://test.com"}]],
        "distances": [[0.5]],
    }
    collection.count.return_value = 5
    # Mock the upsert method for adding documents
    collection.upsert = MagicMock()
    # Mock the get method for content checking
    collection.get = MagicMock(return_value={"ids": [], "metadatas": []})
    return collection


@pytest.fixture
def chroma_handler(mock_client, mock_collection):
    with patch("chromadb.PersistentClient", return_value=mock_client):
        handler = ChromaHandler("test_collection")
        # Mock the chunking manager to return single chunk
        handler._chunking_manager.chunk_text = MagicMock(return_value=["Test content"])
        mock_client.get_or_create_collection.return_value = mock_collection
        return handler


def test_singleton_pattern():
    """Test that ChromaHandler follows singleton pattern"""
    with patch("chromadb.PersistentClient") as mock_client:
        handler1 = ChromaHandler("test1")
        handler2 = ChromaHandler("test2")
        assert handler1 is handler2
        mock_client.assert_called_once_with(
            path=ChromaHandler.get_db_path(),
            settings=Settings(anonymized_telemetry=False),
        )


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
    # Set up mock collection to properly track calls
    mock_collection.upsert = MagicMock()

    # Test adding a document
    url = "http://test.com/page"
    content = "Test content"
    chroma_handler.add_document(content, url)

    # Verify the document was added
    mock_collection.upsert.assert_called()
    call_args = mock_collection.upsert.call_args[1]
    assert "documents" in call_args
    assert len(call_args["documents"]) > 0
    assert call_args["documents"][0] == "Test content"


def test_add_document_github(chroma_handler, mock_collection):
    """Test adding GitHub file content without chunking"""
    # Set up mock collection to properly track calls
    mock_collection.upsert = MagicMock()

    # Test GitHub file URL
    github_url = "https://github.com/owner/repo/blob/main/file.py"
    content = "def hello():\n    return 'world'"

    chroma_handler.add_document(content, github_url)

    # Verify the document was added without chunking
    mock_collection.upsert.assert_called()
    call_args = mock_collection.upsert.call_args[1]
    assert "documents" in call_args
    assert len(call_args["documents"]) > 0
    assert call_args["documents"][0] == content


def test_add_document_regular_vs_github(chroma_handler, mock_collection):
    """Test different handling of regular vs GitHub content"""
    # Set up mock collection to properly track calls
    mock_collection.upsert = MagicMock()

    # Regular web content should be chunked
    web_url = "https://example.com/docs"
    web_content = "A" * 2000  # Long content that would normally be chunked

    # Mock chunking manager to return multiple chunks for web content
    chunks = [web_content[i : i + 500] for i in range(0, len(web_content), 500)]
    chroma_handler._chunking_manager.chunk_text = MagicMock(return_value=chunks)

    chroma_handler.add_document(web_content, web_url)

    # Verify web content was chunked
    mock_collection.upsert.assert_called()
    web_call = mock_collection.upsert.call_args[1]
    assert "documents" in web_call
    assert len(web_call["documents"]) == len(chunks)

    # GitHub content should not be chunked
    github_url = "https://github.com/owner/repo/blob/main/file.py"
    github_content = "A" * 2000

    # Reset mock and add GitHub content
    mock_collection.upsert.reset_mock()
    chroma_handler.add_document(github_content, github_url)

    # Verify GitHub content was not chunked
    mock_collection.upsert.assert_called()
    github_call = mock_collection.upsert.call_args[1]
    assert "documents" in github_call
    assert len(github_call["documents"]) == 1
    assert github_call["documents"][0] == github_content


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

    with patch(
        "chromadb.PersistentClient", return_value=mock_client
    ) as mock_client_class:
        result = ChromaHandler.delete_collection("test_collection")
        assert result is True
        mock_client_class.assert_called_once_with(
            path=ChromaHandler.get_db_path(),
            settings=Settings(anonymized_telemetry=False),
        )
        mock_client.delete_collection.assert_called_once_with("test_collection")


def test_delete_collection_error():
    """Test error handling in delete_collection"""
    mock_client = MagicMock()
    mock_client.delete_collection.side_effect = Exception("Test error")

    with patch("chromadb.PersistentClient", return_value=mock_client):
        result = ChromaHandler.delete_collection("test_collection")
        assert result is False


def test_get_available_collections():
    """Test getting available collections"""
    mock_client = MagicMock()
    mock_client.list_collections.return_value = ["collection1", "collection2"]

    with patch("chromadb.PersistentClient", return_value=mock_client):
        collections = ChromaHandler.get_available_collections()
        assert collections == ["collection1", "collection2"]
        mock_client.list_collections.assert_called_once()


def test_has_matching_content(chroma_handler, mock_collection):
    """Test checking for duplicate content"""
    url = "http://test.com/page"
    content = "Test content"

    # Test when content doesn't exist
    mock_collection.get.return_value = {"ids": [], "metadatas": []}
    assert not chroma_handler.has_matching_content(url, content)

    # Test when content exists but doesn't match
    mock_collection.get.return_value = {
        "ids": ["test_id"],
        "metadatas": [{"url": url, "content_hash": "different_hash"}],
    }
    assert not chroma_handler.has_matching_content(url, content)

    # Test when content exists and matches
    content_hash = blake2b(content.encode(), digest_size=16).hexdigest()
    mock_collection.get.return_value = {
        "ids": ["test_id"],
        "metadatas": [{"url": url, "content_hash": content_hash}],
    }
    assert chroma_handler.has_matching_content(url, content)

    # Verify the correct query was made
    mock_collection.get.assert_called_with(where={"url": url}, include=["metadatas"])


def test_add_document_with_content_tracking(chroma_handler, mock_collection):
    """Test that documents are stored with content hashes"""
    url = "http://test.com/page"
    content = "Test content"

    # Set up mock collection to properly track calls
    mock_collection.upsert = MagicMock()

    # Add document
    chroma_handler.add_document(content, url)

    # Verify content hash was included in metadata
    expected_hash = blake2b(content.encode(), digest_size=16).hexdigest()

    # Verify upsert was called with correct parameters
    mock_collection.upsert.assert_called()
    call_args = mock_collection.upsert.call_args[1]
    assert "metadatas" in call_args
    assert len(call_args["metadatas"]) > 0
    assert call_args["metadatas"][0]["content_hash"] == expected_hash


def test_content_change_detection(chroma_handler, mock_collection):
    """Test the full workflow of content change detection"""
    url = "http://test.com/page"
    original_content = "Original content"
    modified_content = "Modified content"

    # Set up mock collection to properly track calls
    mock_collection.upsert = MagicMock()

    # First addition
    chroma_handler.add_document(original_content, url)

    # Setup mock to return the original content's metadata
    original_hash = blake2b(original_content.encode(), digest_size=16).hexdigest()
    mock_collection.get.return_value = {
        "ids": ["test_id"],
        "metadatas": [{"url": url, "content_hash": original_hash}],
    }

    # Check with same content
    assert chroma_handler.has_matching_content(url, original_content)

    # Check with modified content
    assert not chroma_handler.has_matching_content(url, modified_content)

    # Add modified content
    chroma_handler.add_document(modified_content, url)

    # Verify the document was updated with new hash
    modified_hash = blake2b(modified_content.encode(), digest_size=16).hexdigest()

    # Verify upsert was called with correct parameters
    mock_collection.upsert.assert_called()
    last_call = mock_collection.upsert.call_args[1]
    assert "metadatas" in last_call
    assert len(last_call["metadatas"]) > 0
    assert last_call["metadatas"][0]["content_hash"] == modified_hash
