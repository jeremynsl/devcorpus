import pytest
import os
import json
from unittest.mock import patch, MagicMock, call
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from scraper_chat.embeddings.embeddings import Reranker, EmbeddingManager
from sentence_transformers import CrossEncoder


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return {
        "embeddings": {
            "models": {
                "available": [
                    "model1",
                    "model2",
                    "model3"
                ],
                "default": "model1",
                "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2"
            }
        },
        "pytorch_device": "cpu"  # Explicitly set device to cpu for tests
    }


@pytest.fixture
def mock_config_file(tmp_path, mock_config):
    """Create a temporary config file for testing."""
    config_path = tmp_path / "scraper_config.json"
    with open(config_path, "w") as f:
        json.dump(mock_config, f)

    # Temporarily replace the config file path and os.path.dirname
    with (
        patch("scraper_chat.embeddings.embeddings.CONFIG_FILE", str(config_path)),
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


def test_embedding_manager_singleton():
    """Test that EmbeddingManager follows the singleton pattern."""
    manager1 = EmbeddingManager()
    manager2 = EmbeddingManager()

    # Verify they are the same instance
    assert manager1 is manager2


def test_embedding_model_switching(mock_config_file):
    """Test switching between different embedding models."""
    with patch("scraper_chat.embeddings.embeddings.embedding_functions.SentenceTransformerEmbeddingFunction") as MockEmbeddingFunction:
        # Configure the mock embedding functions
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        
        def create_mock_model(model_name, device):
            if model_name == "model1":
                return mock_model1
            elif model_name == "model2":
                return mock_model2
            
        MockEmbeddingFunction.side_effect = create_mock_model
        
        # Reset singleton instance
        EmbeddingManager._instance = None
        EmbeddingManager._embedding_function = None
        EmbeddingManager._current_model = None
        
        # Initialize with default model (model1)
        manager = EmbeddingManager()
        assert manager._current_model == "model1"
        assert manager._embedding_function is mock_model1
        
        # Update config to use model2
        config = {
            "embeddings": {
                "models": {
                    "available": ["model1", "model2", "model3"],
                    "default": "model2"
                }
            },
            "pytorch_device": "cpu"
        }
        with open(mock_config_file, "w") as f:
            json.dump(config, f)
            
        # Reset singleton to force reinitialization
        EmbeddingManager._instance = None
        
        # Create new instance with updated config
        manager = EmbeddingManager()
        assert manager._current_model == "model2"
        assert manager._embedding_function is mock_model2
        
        # Verify the embedding functions were created with correct parameters
        MockEmbeddingFunction.assert_has_calls([
            call(model_name="model1", device="cpu"),
            call(model_name="model2", device="cpu")
        ])


def test_embedding_model_fallback(mock_config_file):
    """Test that EmbeddingManager falls back to available models on error."""
    with patch("scraper_chat.embeddings.embeddings.embedding_functions.SentenceTransformerEmbeddingFunction") as MockEmbeddingFunction:
        # Configure mock to fail for model1 but succeed for model2
        mock_model2 = MagicMock()
        def create_mock_model(model_name, device):
            if model_name == "model1":
                raise Exception("Model load error")
            elif model_name == "model2":
                return mock_model2
            
        MockEmbeddingFunction.side_effect = create_mock_model
        
        # Create config with multiple models
        config = {
            "embeddings": {
                "models": {
                    "available": ["model1", "model2", "model3"],
                    "default": "model1"
                }
            },
            "pytorch_device": "cpu"
        }
        with open(mock_config_file, "w") as f:
            json.dump(config, f)
        
        # Reset singleton instance
        EmbeddingManager._instance = None
        EmbeddingManager._embedding_function = None
        EmbeddingManager._current_model = None
        
        # Initialize manager - should fall back to model2
        manager = EmbeddingManager()
        
        # Verify fallback behavior
        assert manager._current_model == "model2"
        assert manager._embedding_function is mock_model2
        
        # Verify both attempts were made with correct parameters
        MockEmbeddingFunction.assert_has_calls([
            call(model_name="model1", device="cpu"),  # First attempt with default model
            call(model_name="model2", device="cpu")   # Successful fallback
        ])


def test_reranker_with_empty_documents():
    """Test reranking with empty documents."""
    reranker = Reranker()
    original_model = reranker._rerank_model
    reranker._rerank_model = MagicMock()

    # Test with empty document list
    ranked_indices = reranker.rerank("test query", [], top_k=5)
    assert ranked_indices == []

    # Test with None model
    reranker._rerank_model = None
    ranked_indices = reranker.rerank("test query", ["doc1", "doc2"], top_k=1)
    assert ranked_indices == [0]

    # Restore the original model
    reranker._rerank_model = original_model


def test_reranker_with_no_model(mock_config_file):
    """Test reranking when no model is configured."""
    # Create config without reranker model
    config = {
        "embeddings": {
            "models": {
                "available": ["model1", "model2"],
                "default": "model1"
            }
        }
    }
    with open(mock_config_file, "w") as f:
        json.dump(config, f)
    
    # Reset singleton to force reinitialization
    Reranker._instance = None
    Reranker._rerank_model = None
    
    # Create new instance
    reranker = Reranker()
    
    # Verify no model was loaded
    assert reranker._rerank_model is None
    
    # Verify fallback behavior
    ranked_indices = reranker.rerank("test query", ["doc1", "doc2", "doc3"], top_k=2)
    assert ranked_indices == [0, 1]  # Should return original order
