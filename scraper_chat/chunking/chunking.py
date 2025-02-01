"""Text chunking module."""

from chunking_evaluation.chunking import ClusterSemanticChunker, RecursiveTokenChunker
import logging
from scraper_chat.config.config import load_config, CONFIG_FILE
from threading import Lock

logger = logging.getLogger(__name__)


class ChunkingManager:
    _instance = None
    _lock = Lock()

    def __new__(cls) -> "ChunkingManager":
        """Singleton pattern to ensure one chunker instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ChunkingManager, cls).__new__(cls)
                    # Default to using RecursiveTokenChunker, but allow switching later.
                    cls._instance._initialize(
                        use_recursive=True
                    )  # Ensure consistency with chosen default
        return cls._instance

    def _initialize(self, use_recursive: bool = False) -> None:
        """Initialize the chunker with default settings."""
        # Load chunking configuration
        config = load_config(CONFIG_FILE)
        chunking_config = config.get("chunking", {})
        chunk_size = chunking_config.get("chunk_size", 200)
        max_chunk_size = chunking_config.get("max_chunk_size", 200)
        chunk_overlap = chunking_config.get("chunk_overlap", 0)

        if use_recursive:
            # Initialize RecursiveTokenChunker with config settings
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
        """Switch to using RecursiveTokenChunker at runtime."""
        # Load config values if not provided in kwargs
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
        """Switch to using ClusterSemanticChunker at runtime."""
        with self._lock:
            from scraper_chat.embeddings.embeddings import EmbeddingManager

            # Use provided embedding_function or get default
            ef = kwargs.pop("embedding_function", EmbeddingManager().embedding_function)

            # Load config values if not provided in kwargs
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
        """Chunk the input text into semantic chunks."""

        if not self._chunker:
            raise ValueError("Chunker not initialized.")
        if not text.strip():
            return []
        return self._chunker.split_text(text)

    def get_chunking_method(self) -> str:
        """Get the current chunking method name."""
        return self._chunking_method
