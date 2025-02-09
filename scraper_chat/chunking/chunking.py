"""
Text chunking module for splitting documents into semantically meaningful segments.
Supports both token-based recursive chunking and semantic clustering approaches.
Configuration is loaded from the global config file and can be overridden at runtime.
"""

from chunking_evaluation.chunking import ClusterSemanticChunker, RecursiveTokenChunker
import logging
from scraper_chat.config.config import load_config, CONFIG_FILE
from threading import Lock

logger = logging.getLogger(__name__)


class ChunkingManager:
    """
    Singleton manager for text chunking operations.
    Provides thread-safe access to chunking functionality with configurable methods.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls) -> "ChunkingManager":
        """
        Create or return the singleton instance of ChunkingManager.
        Thread-safe implementation using double-checked locking pattern.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ChunkingManager, cls).__new__(cls)
                    cls._instance._initialize(use_recursive=True)
        return cls._instance

    def _initialize(self, use_recursive: bool = False) -> None:
        """
        Initialize the chunking engine with configuration settings.

        Args:
            use_recursive: If True, use RecursiveTokenChunker; otherwise use ClusterSemanticChunker
        """
        config = load_config(CONFIG_FILE)
        chunking_config = config.get("chunking", {})
        chunk_size = chunking_config.get("chunk_size", 200)
        max_chunk_size = chunking_config.get("max_chunk_size", 200)
        chunk_overlap = chunking_config.get("chunk_overlap", 0)

        if use_recursive:
            self._chunker = RecursiveTokenChunker(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            self._chunking_method = "RecursiveTokenChunker"
        else:
            from scraper_chat.embeddings.embeddings import EmbeddingManager

            default_ef = EmbeddingManager().embedding_function
            self._chunker = ClusterSemanticChunker(
                default_ef, max_chunk_size=max_chunk_size
            )
            self._chunking_method = "ClusterSemanticChunker"

    def use_recursive_chunker(self, **kwargs) -> None:
        """
        Switch to RecursiveTokenChunker with optional configuration override.

        Args:
            **kwargs: Optional configuration overrides for chunk_size and chunk_overlap.
                     If not provided, values are loaded from config file.
        """
        with self._lock:
            if not kwargs:
                config = load_config(CONFIG_FILE)
                chunking_config = config.get("chunking", {})
                kwargs = {
                    "chunk_size": chunking_config.get("chunk_size", 200),
                    "chunk_overlap": chunking_config.get("chunk_overlap", 0),
                }
            self._chunker = RecursiveTokenChunker(**kwargs)
            self._chunking_method = "RecursiveTokenChunker"

    def use_cluster_chunker(self, **kwargs) -> None:
        """
        Switch to ClusterSemanticChunker with optional configuration override.

        Args:
            **kwargs: Optional configuration overrides including:
                     - embedding_function: Custom embedding function
                     - max_chunk_size: Maximum size for each chunk
                     If not provided, values are loaded from config file.
        """
        with self._lock:
            from scraper_chat.embeddings.embeddings import EmbeddingManager

            ef = kwargs.pop("embedding_function", EmbeddingManager().embedding_function)

            if "chunk_size" not in kwargs:
                config = load_config(CONFIG_FILE)
                chunking_config = config.get("chunking", {})
                kwargs = {
                    "max_chunk_size": chunking_config.get("max_chunk_size", 200),
                    **kwargs,
                }
            self._chunker = ClusterSemanticChunker(ef, **kwargs)
            self._chunking_method = "ClusterSemanticChunker"

    def chunk_text(self, text: str) -> list[str]:
        """
        Split input text into semantically meaningful chunks.

        Args:
            text: Input text to be chunked

        Returns:
            List of text chunks

        Raises:
            ValueError: If chunker is not initialized or text is empty
        """
        if not self._chunker:
            raise ValueError("Chunker not initialized.")
        if not text.strip():
            return []
        return self._chunker.split_text(text)

    def get_chunking_method(self) -> str:
        """
        Get the name of the currently active chunking method.

        Returns:
            String name of the current chunking method
        """
        return self._chunking_method
