import pytest
import os
import json
from unittest.mock import patch, MagicMock, call
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from scraper_chat.embeddings.embeddings import Reranker
from sentence_transformers import CrossEncoder


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return {
        "embeddings": {"reranker": {"model": "cross-encoder/ms-marco-MiniLM-L-6-v2"}}
    }


@pytest.fixture
def mock_config_file(tmp_path, mock_config):
    """Create a temporary config file for testing."""
    config_path = tmp_path / "scraper_config.json"
    with open(config_path, "w") as f:
        json.dump(mock_config, f)

    # Temporarily replace the config file path
    with (
        patch("scraper_chat.embeddings.embeddings.CONFIG_FILE", config_path.name),
        patch("os.path.dirname", return_value=str(tmp_path)),
    ):
        yield config_path


def test_reranker_singleton():
    """Test that Reranker follows the singleton pattern."""
    reranker1 = Reranker()
    reranker2 = Reranker()

    # Verify they are the same instance
    assert reranker1 is reranker2


def test_reranker_initialization(mock_config_file):
    """Test the initialization of the Reranker."""
    reranker = Reranker()

    # Verify the reranking model is initialized
    assert reranker._rerank_model is not None

    # Verify the model name is as expected
    assert isinstance(reranker._rerank_model, CrossEncoder)


def test_reranker_with_mocked_cross_encoder(mock_config_file):
    """Test the rerank method with mocked CrossEncoder."""
    # Prepare test data
    query = "test query"
    documents = [
        "First document about something",
        "Second document about another thing",
        "Third document with different content",
    ]

    # Create a mock CrossEncoder
    with patch("scraper_chat.embeddings.embeddings.CrossEncoder") as MockCrossEncoder:
        # Configure the mock to return predefined scores
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict.return_value = [0.9, 0.7, 0.5]
        MockCrossEncoder.return_value = mock_cross_encoder

        # Reinitialize Reranker to use the mocked CrossEncoder
        # Reset the singleton instance to force reinitialization
        Reranker._instance = None
        Reranker._rerank_model = None

        # Create a new Reranker instance
        reranker = Reranker()

        # Verify the CrossEncoder was initialized
        MockCrossEncoder.assert_called_once()

        # Rerank documents
        ranked_indices = reranker.rerank(query, documents, top_k=2)

        # Verify method calls
        mock_cross_encoder.predict.assert_called_once_with(
            [
                [query, "First document about something"],
                [query, "Second document about another thing"],
                [query, "Third document with different content"],
            ]
        )

        # Verify returned indices are correct
        assert len(ranked_indices) == 2
        assert ranked_indices[0] == 0  # First document should be top-ranked
        assert ranked_indices[1] == 1  # Second document should be second


def test_reranker_with_empty_documents():
    """Test reranking with empty documents."""
    reranker = Reranker()
    query = "test query"
    documents = []

    # Rerank empty documents
    ranked_indices = reranker.rerank(query, documents)

    # Verify empty list returns empty list
    assert len(ranked_indices) == 0


def test_reranker_with_no_model():
    """Test reranking when no model is available."""
    reranker = Reranker()

    # Temporarily remove the rerank model
    original_model = reranker._rerank_model
    reranker._rerank_model = None

    query = "test query"
    documents = ["doc1", "doc2", "doc3"]

    # Rerank with no model
    ranked_indices = reranker.rerank(query, documents, top_k=2)

    # Verify it returns original indices
    assert len(ranked_indices) == 2
    assert ranked_indices == [0, 1]

    # Restore the original model
    reranker._rerank_model = original_model
