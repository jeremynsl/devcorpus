"""
Embeddings and reranking module for semantic search.
Provides thread-safe singleton access to:
- SentenceTransformer embedding models for vector encoding
- Cross-encoder models for result reranking
- Automatic fallback handling for model loading
"""

from chromadb.utils import embedding_functions
import logging
from sentence_transformers import CrossEncoder
from scraper_chat.config.config import load_config, CONFIG_FILE
from threading import Lock
from typing import Union

logger = logging.getLogger(__name__)


class Reranker:
    """
    Thread-safe singleton for reranking search results using cross-encoder models.
    Loads model configuration from config file with fallback to original order.
    """

    _instance = None
    _rerank_model = None
    _lock = Lock()

    def __new__(cls) -> "Reranker":
        """
        Get or create singleton instance.
        Thread-safe implementation using double-checked locking.

        Returns:
            Reranker instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Reranker, cls).__new__(cls)
                    cls._instance._initialize_reranker()
        return cls._instance

    def _initialize_reranker(self) -> None:
        """
        Initialize reranking model from config.
        Uses device specified in config (defaults to CPU).
        Logs warning if no reranker is configured.
        """
        try:
            config = load_config(CONFIG_FILE)
            model_name = config.get("embeddings", {}).get("models", {}).get("reranker")
            if not model_name:
                logger.warning(f"No reranker configured in {CONFIG_FILE}")
                return

            device = config.get("pytorch_device", "cpu")
            logger.info(
                f"Initializing reranking model: {model_name} on device: {device}"
            )
            self._rerank_model = CrossEncoder(model_name, device=device)

        except Exception as e:
            logger.exception(f"Error loading reranking model: {e}")
            self._rerank_model = None

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
        return_scores: bool = True,
    ) -> Union[list[int], tuple[list[int], list[float]]]:
        """
        Rerank documents by relevance to query using cross-encoder scoring.

        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top results to return
            return_scores: If True, return tuple of (indices, scores), otherwise just indices

        Returns:
            If return_scores is True:
                Tuple of (indices, scores) where:
                - indices: List of indices for top-k documents sorted by relevance
                - scores: List of corresponding similarity scores (higher is better)
            If return_scores is False:
                List of indices for top-k documents sorted by relevance
            Falls back to original order if reranking fails.
        """
        if not self._rerank_model or not documents:
            indices = list(range(min(top_k, len(documents))))
            if return_scores:
                return indices, [None] * len(indices)
            return indices

        try:
            scores = self._rerank_model.predict([[query, doc] for doc in documents])
            ranked_pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            top_pairs = ranked_pairs[:top_k]
            indices = [i for i, _ in top_pairs]
            if return_scores:
                scores = [s for _, s in top_pairs]
                return indices, scores
            return indices
        except Exception as e:
            logger.exception(f"Error during reranking: {e}")
            indices = list(range(min(top_k, len(documents))))
            if return_scores:
                return indices, [None] * len(indices)
            return indices


class EmbeddingManager:
    """
    Thread-safe singleton for managing document embedding models.
    Features:
    - Automatic model initialization from config
    - Fallback to alternative models if primary fails
    - Device configuration (CPU/GPU)
    - Caching of model instances
    """

    _instance = None
    _embedding_function = None
    _lock = Lock()
    _current_model = None

    def __new__(cls) -> "EmbeddingManager":
        """
        Get or create singleton instance.
        Thread-safe implementation using double-checked locking.

        Returns:
            EmbeddingManager instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(EmbeddingManager, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """
        Initialize embedding model from config.

        Attempts to load the default model specified in config.
        If default model fails, attempts to load alternative models
        from the available models list. Falls back to CPU if GPU fails.
        """
        try:
            config = load_config(CONFIG_FILE)
            logger.info(f"Loaded config from {CONFIG_FILE}")

            model_name = config.get("embeddings", {}).get("models", {}).get("default")
            if not model_name:
                logger.error(f"No default embedding model configured in {CONFIG_FILE}")
                return

            device = config.get("pytorch_device", "cpu")
            logger.info(
                f"Initializing embedding model: {model_name} on device: {device}"
            )

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

        if not self._embedding_function:
            logger.error("Failed to initialize any embedding model")

    @property
    def embedding_function(self) -> embedding_functions.EmbeddingFunction:
        """
        Get current embedding function.

        Returns:
            SentenceTransformer embedding function instance

        Note:
            May return None if initialization failed
        """
        return self._embedding_function
