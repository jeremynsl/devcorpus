"""Embedding model management module."""

import json
import os
from chromadb.utils import embedding_functions
import logging

logger = logging.getLogger(__name__)

CONFIG_FILE = "scraper_config.json"

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
            config_path = os.path.join(os.path.dirname(__file__), CONFIG_FILE)
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            model_name = config.get('embeddings', {}).get('models', {}).get('default')
            if not model_name:
                model_name = "avsolatorio/GIST-Embedding-v0"  # Fallback to maintain compatibility
                logger.warning(f"No embedding model configured, using default: {model_name}")
            
            logger.info(f"Initializing embedding model: {model_name}")
            self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
            
        except Exception as e:
            logger.error(f"Error loading embedding model, using default: {e}")
            # Fallback to maintain compatibility
            self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="avsolatorio/GIST-Embedding-v0"
            )
    
    @property
    def embedding_function(self):
        """Get the current embedding function."""
        return self._embedding_function
