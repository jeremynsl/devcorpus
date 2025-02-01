"""Embedding model management module."""

from chromadb.utils import embedding_functions
import logging
from sentence_transformers import CrossEncoder
from scraper_chat.config.config import load_config, CONFIG_FILE
from threading import Lock
logger = logging.getLogger(__name__)


# Add to imports
class ConfigDefaults:
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    EMBEDDING_MODEL = "avsolatorio/GIST-Embedding-v0"

class Reranker:
    _instance = None
    _rerank_model = None
    _lock = Lock()
    def __new__(cls) -> "Reranker":
        """Singleton pattern for reranking model"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Reranker, cls).__new__(cls)
                    cls._instance._initialize_reranker()
        return cls._instance

    def _initialize_reranker(self) -> None:
        """Initialize reranking model from config"""
        try:
            config = load_config(CONFIG_FILE)

            model_name = config.get("embeddings", {}).get("reranker", {}).get("model")
            if not model_name:
                model_name = ConfigDefaults.RERANKER_MODEL  # Default reranker
                logger.warning(f"No reranker configured, using default: {model_name}")

            logger.info(f"Initializing reranking model: {model_name}")
            self._rerank_model = CrossEncoder(model_name)

        except Exception as e:
            logger.exception(f"Error loading reranking model: {e}")
            self._rerank_model = None

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[int]:
        """Rerank documents using cross-encoder"""
        if not self._rerank_model or not documents:
            return list(range(min(top_k, len(documents))))  # Fallback to original order

        # Create pairs for scoring
        pairs = [[query, doc] for doc in documents]

        # Get scores from cross-encoder
        scores = self._rerank_model.predict(pairs)

        # Sort by descending scores
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )

        return ranked_indices[:top_k]


class EmbeddingManager:
    _instance = None
    _embedding_function = None
    _lock = Lock()

    def __new__(cls) -> "EmbeddingManager":
        """Singleton pattern to ensure one embedding model instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(EmbeddingManager, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize the embedding model from config."""
        try:
            config = load_config(CONFIG_FILE)

            model_name = config.get("embeddings", {}).get("models", {}).get("default")
            if not model_name:
                model_name = ConfigDefaults.EMBEDDING_MODEL  # Fallback to maintain compatibility
                logger.warning(
                    f"No embedding model configured, using default: {model_name}"
                )

            logger.info(f"Initializing embedding model: {model_name}")
            self._embedding_function = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=model_name
                )
            )

        except Exception as e:
            logger.exception(f"Error loading embedding model, using default: {e}")
            # Fallback to maintain compatibility
            self._embedding_function = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=ConfigDefaults.EMBEDDING_MODEL
                )
            )

    @property
    def embedding_function(self) -> embedding_functions.EmbeddingFunction:
        """Get the current embedding function."""
        return self._embedding_function
