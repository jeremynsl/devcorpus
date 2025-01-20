"""Test chunking functionality."""

import pytest
from unittest.mock import patch
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

# Create a proper mock of ClusterSemanticChunker
class MockClusterSemanticChunker:
    def __init__(self, embedding_function=None, max_chunk_size=400, min_chunk_size=50, length_function=None):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.embedding_function = embedding_function
        
    def split_text(self, text: str) -> list[str]:
        # Simulate the chunking behavior based on max_chunk_size
        if not text.strip():
            return []
        # Return two chunks to simulate clustering
        return [
            "First sentence in cluster one. Second sentence also in cluster one.",
            "First sentence in cluster two. Second sentence also in cluster two."
        ]

# Import normally without patch
from chunking import ChunkingManager
from chunking_evaluation.chunking import RecursiveTokenChunker

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the ChunkingManager singleton between tests."""
    ChunkingManager._instance = None
    yield
    ChunkingManager._instance = None

def test_chunking_manager_singleton():
    """Test that ChunkingManager maintains singleton pattern."""
    manager1 = ChunkingManager()
    manager2 = ChunkingManager()
    assert manager1 is manager2

@pytest.mark.parametrize("use_recursive", [True, False])
def test_chunking_manager_initialization(use_recursive):
    """Test that ChunkingManager initializes with correct chunker type."""
    with patch('chunking.ClusterSemanticChunker', MockClusterSemanticChunker):
        manager = ChunkingManager()
        manager._initialize(use_recursive=use_recursive)
        
        if use_recursive:
            assert isinstance(manager._chunker, RecursiveTokenChunker)
            assert manager._chunking_method == "RecursiveTokenChunker"
        else:
            assert isinstance(manager._chunker, MockClusterSemanticChunker)
            assert manager._chunking_method == "ClusterSemanticChunker"

def test_recursive_chunking():
    """Test RecursiveTokenChunker functionality."""
    manager = ChunkingManager()
    manager.use_recursive_chunker(chunk_size=100, chunk_overlap=0)
    
    # Test with a simple paragraph
    text = "This is a test paragraph. " * 10  # About 200 characters
    chunks = manager.chunk_text(text)
    
    assert len(chunks) > 1  # Should split into at least 2 chunks
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(len(chunk) > 0 for chunk in chunks)
    # Rough check that chunks are around the right size
    assert all(len(chunk) <= 150 for chunk in chunks)  # Allow some flexibility

@pytest.mark.asyncio
async def test_cluster_chunking():
    """Test ClusterSemanticChunker functionality."""
    with patch('chunking.ClusterSemanticChunker', MockClusterSemanticChunker):
        manager = ChunkingManager()
        manager.use_cluster_chunker(max_chunk_size=2)
        
        text = (
            "First sentence in cluster one. "
            "Second sentence also in cluster one. "
            "First sentence in cluster two. "
            "Second sentence also in cluster two."
        )
        chunks = manager.chunk_text(text)
        
        # Verify clustering behavior
        assert len(chunks) == 2
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk) > 0 for chunk in chunks)
        assert "cluster one" in chunks[0]
        assert "cluster two" in chunks[1]

def test_empty_text():
    """Test handling of empty text."""
    manager = ChunkingManager()
    
    # Test with both chunker types
    manager.use_recursive_chunker()
    assert manager.chunk_text("") == []
    assert manager.chunk_text("   ") == []
    
    manager.use_cluster_chunker()
    assert manager.chunk_text("") == []
    assert manager.chunk_text("   ") == []

def test_chunking_method_tracking():
    """Test that chunking method is correctly tracked when switching."""
    manager = ChunkingManager()
    
    manager.use_recursive_chunker()
    assert manager.get_chunking_method() == "RecursiveTokenChunker"
    
    manager.use_cluster_chunker()
    assert manager.get_chunking_method() == "ClusterSemanticChunker"

def test_custom_parameters():
    """Test chunkers with custom parameters."""
    manager = ChunkingManager()
    
    # Test RecursiveTokenChunker with custom parameters
    manager.use_recursive_chunker(chunk_size=50, chunk_overlap=10)
    text = "Short text for testing custom parameters."
    chunks = manager.chunk_text(text)
    assert isinstance(chunks, list)
    
    # Test ClusterSemanticChunker with custom parameters
    manager.use_cluster_chunker(max_chunk_size=100, min_chunk_size=25)
    chunks = manager.chunk_text(text)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
