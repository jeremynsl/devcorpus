"""Embedding model management module."""

from chromadb.utils import embedding_functions
import logging
from sentence_transformers import CrossEncoder
from scraper_chat.config.config import load_config, CONFIG_FILE
from threading import Lock

logger = logging.getLogger(__name__)


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

            model_name = config.get("embeddings", {}).get("models", {}).get("reranker")
            if not model_name:
                logger.warning(f"No reranker configured in {CONFIG_FILE}")
                return

            # Get device from config, default to cpu
            device = config.get("pytorch_device", "cpu")
            logger.info(
                f"Initializing reranking model: {model_name} on device: {device}"
            )
            self._rerank_model = CrossEncoder(model_name, device=device)

        except Exception as e:
            logger.exception(f"Error loading reranking model: {e}")
            self._rerank_model = None

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[int]:
        """Rerank documents using cross-encoder"""
        if not self._rerank_model or not documents:
            return list(range(min(top_k, len(documents))))  # Fallback to original order

        try:
            # Use lists instead of tuples for compatibility with test expectations
            scores = self._rerank_model.predict([[query, doc] for doc in documents])
            ranked_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )
            return ranked_indices[:top_k]
        except Exception as e:
            logger.exception(f"Error during reranking: {e}")
            return list(range(min(top_k, len(documents))))  # Fallback to original order


class EmbeddingManager:
    _instance = None
    _embedding_function = None
    _lock = Lock()
    _current_model = None

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
            logger.info(f"Loaded config from {CONFIG_FILE}")

            # Get model name from config
            model_name = config.get("embeddings", {}).get("models", {}).get("default")
            if not model_name:
                logger.error(f"No default embedding model configured in {CONFIG_FILE}")
                return

            # Get device from config, default to cpu
            device = config.get("pytorch_device", "cpu")
            logger.info(
                f"Initializing embedding model: {model_name} on device: {device}"
            )

            # Only reinitialize if model has changed
            if model_name != self._current_model:
                try:
                    self._embedding_function = (
                        embedding_functions.SentenceTransformerEmbeddingFunction(
                            model_name=model_name, device=device
                        )
                    )
                    self._current_model = model_name
                    logger.info(f"Successfully initialized model: {model_name}")
                except Exception as e:
                    logger.exception(f"Error loading model {model_name}: {e}")
                    # Try to load any available model from the config
                    available_models = (
                        config.get("embeddings", {})
                        .get("models", {})
                        .get("available", [])
                    )
                    for fallback_model in available_models:
                        if fallback_model != model_name:
                            try:
                                logger.info(
                                    f"Attempting to load fallback model: {fallback_model}"
                                )
                                self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                                    model_name=fallback_model,
                                    device="cpu",  # Always use CPU for fallback
                                )
                                self._current_model = fallback_model
                                logger.info(
                                    f"Successfully loaded fallback model: {fallback_model}"
                                )
                                break
                            except Exception as e2:
                                logger.warning(
                                    f"Failed to load fallback model {fallback_model}: {e2}"
                                )
                                continue

        except Exception as e:
            logger.exception(f"Error during embedding model initialization: {e}")

        # If we still don't have a working model, log an error
        if not self._embedding_function:
            logger.error("Failed to initialize any embedding model")

    @property
    def embedding_function(self) -> embedding_functions.EmbeddingFunction:
        """Get the current embedding function."""
        return self._embedding_function
