"""Text chunking module."""

from chunking_evaluation.chunking import ClusterSemanticChunker, RecursiveTokenChunker

class ChunkingManager:
    _instance = None
    _chunker = None
    _chunking_method = None
    
    def __new__(cls):
        """Singleton pattern to ensure one chunker instance."""
        if cls._instance is None:
            cls._instance = super(ChunkingManager, cls).__new__(cls)
            # Default to using ClusterSemanticChunker, but allow switching later.
            cls._instance._initialize(use_recursive=True)  # Default to recursive chunking
        return cls._instance
    
    def _initialize(self, use_recursive: bool = False):
        """Initialize the chunker with default settings."""
        if use_recursive:
            # Initialize RecursiveTokenChunker with desired settings.
            self._chunker = RecursiveTokenChunker(chunk_size=200, chunk_overlap=0)
            self._chunking_method = "RecursiveTokenChunker"
        else:
            from embeddings import EmbeddingManager
            default_ef = EmbeddingManager().embedding_function
            self._chunker = ClusterSemanticChunker(default_ef, max_chunk_size=400)
            self._chunking_method = "ClusterSemanticChunker"
    
    def use_recursive_chunker(self, **kwargs):
        """Switch to using RecursiveTokenChunker at runtime."""
        self._chunker = RecursiveTokenChunker(**kwargs)
        self._chunking_method = "RecursiveTokenChunker"
    
    def use_cluster_chunker(self, **kwargs):
        """Switch to using ClusterSemanticChunker at runtime."""
        from embeddings import EmbeddingManager
        # Use provided embedding_function or get default
        ef = kwargs.pop('embedding_function', EmbeddingManager().embedding_function)
        self._chunker = ClusterSemanticChunker(ef, **kwargs)
        self._chunking_method = "ClusterSemanticChunker"
    
    def chunk_text(self, text: str) -> list[str]:
        """Chunk the input text into semantic chunks."""
        if not text.strip():
            return []
        return self._chunker.split_text(text)
    
    def get_chunking_method(self) -> str:
        """Return the current chunking method."""
        return self._chunking_method
