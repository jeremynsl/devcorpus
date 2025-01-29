"""Embedding model management module."""

from chromadb.utils import embedding_functions
import logging
from sentence_transformers import CrossEncoder
from scraper_chat.config.config import load_config, CONFIG_FILE

logger = logging.getLogger(__name__)


# Add to imports


class Reranker:
    _instance = None
    _rerank_model = None

    def __new__(cls):
        """Singleton pattern for reranking model"""
        if cls._instance is None:
            cls._instance = super(Reranker, cls).__new__(cls)
            cls._instance._initialize_reranker()
        return cls._instance

    def _initialize_reranker(self):
        """Initialize reranking model from config"""
        try:
            config = load_config(CONFIG_FILE)

            model_name = config.get("embeddings", {}).get("reranker", {}).get("model")
            if not model_name:
                model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Default reranker
                logger.warning(f"No reranker configured, using default: {model_name}")

            logger.info(f"Initializing reranking model: {model_name}")
            self._rerank_model = CrossEncoder(model_name)

        except Exception as e:
            logger.error(f"Error loading reranking model: {e}")
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

    def __new__(cls):
        """Singleton pattern to ensure one embedding model instance."""
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the embedding model from config."""
        try:
            config = load_config(CONFIG_FILE)

            model_name = config.get("embeddings", {}).get("models", {}).get("default")
            if not model_name:
                model_name = "avsolatorio/GIST-Embedding-v0"  # Fallback to maintain compatibility
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
            logger.error(f"Error loading embedding model, using default: {e}")
            # Fallback to maintain compatibility
            self._embedding_function = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="avsolatorio/GIST-Embedding-v0"
                )
            )

    @property
    def embedding_function(self):
        """Get the current embedding function."""
        return self._embedding_function
