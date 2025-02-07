import chromadb
from chromadb.config import Settings
import logging
from urllib.parse import urlparse
from scraper_chat.embeddings.embeddings import EmbeddingManager, Reranker
from scraper_chat.chunking.chunking import ChunkingManager
from scraper_chat.config.config import load_config, CONFIG_FILE
from typing import TypedDict
from threading import Lock
from hashlib import blake2b
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


# Default DB path, can be overridden for testing
DEFAULT_DB_PATH = "docs_database.db"


class QueryResult(TypedDict):
    url: str
    text: str
    distance: float
    metadata: dict


class ChromaHandler:
    """Handler for ChromaDB operations with singleton pattern"""

    _instance = None
    _client = None
    _collections = {}
    _embedding_manager = EmbeddingManager()
    _chunking_manager = ChunkingManager()
    _db_path = None
    _lock = Lock()

    @classmethod
    def configure(cls, db_path: str = None):
        """Configure ChromaDB with custom settings"""
        if db_path:
            cls._db_path = db_path
            # Reset instance to force recreation with new path
            cls._instance = None
            cls._client = None
            cls._collections = {}

    @classmethod
    def get_db_path(cls):
        """Get the current DB path"""
        return cls._db_path or DEFAULT_DB_PATH

    def get_database_size(cls) -> int:
        """Get the current database size in bytes"""
        db_path = Path(cls.get_db_path())
        return sum(f.stat().st_size for f in db_path.glob("**/*") if f.is_file())

    @classmethod
    def reset(cls):
        """Reset the singleton instance - primarily for testing"""
        cls._instance = None
        cls._client = None
        cls._collections = {}
        cls._db_path = None

    def __new__(cls, collection_name: str = None):
        """Singleton pattern to ensure one database connection."""
        with cls._lock:
            if cls._instance is None:
                instance = super(ChromaHandler, cls).__new__(cls)
                instance._initialize()
                cls._instance = instance
        return cls._instance

    def _initialize(self):
        """Initialize the ChromaDB client"""
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        if not ChromaHandler._client:
            ChromaHandler._client = chromadb.PersistentClient(
                path=self.get_db_path(), settings=Settings(anonymized_telemetry=False)
            )

        self._initialized = True

    def __init__(self, collection_name: str = None):
        """
        Initialize ChromaDB collection.
        """
        if collection_name and collection_name not in self._collections:
            self._collections[collection_name] = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._embedding_manager.embedding_function,
                metadata={"description": "Scraped website content"},
            )

    def get_collection(self, collection_name: str) -> chromadb.Collection:
        """Get or create a collection by name."""
        if collection_name not in self._collections:
            self._collections[collection_name] = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._embedding_manager.embedding_function,
            )
        return self._collections[collection_name]

    def add_document(self, text: str, url: str) -> None:
        """
        Add a document to the collection with its URL as the ID.
        Handles duplicate URLs by using upsert.
        """
        if not text.strip():
            return
        if not urlparse(url).scheme or not urlparse(url).netloc:
            raise ValueError(f"Invalid URL: {url}")

        # Get collection name from URL
        collection_name = self.get_collection_name(url)
        collection = self.get_collection(collection_name)

        doc_hash = blake2b(url.encode(), digest_size=8).hexdigest()
        doc_id = f"{doc_hash}"

        # Generate content hash
        content_hash = blake2b(text.encode(), digest_size=16).hexdigest()

        # Check for existing metadata
        existing_metadata = {}
        try:
            results = collection.get(ids=[doc_id])
            if results and results["metadatas"]:
                existing_metadata = results["metadatas"][0]
        except:
            pass

        # Merge with new metadata, preserving existing fields
        metadata = {
            **existing_metadata,
            "url": url,
            "content_hash": content_hash,
            "last_updated": datetime.now().isoformat(),
        }

        # Skip chunking for GitHub files, use full file content
        if "github.com" in url and "/blob/" in url:
            chunks = [text]
            chunk_ids = [doc_id]
            chunk_metadatas = [metadata]
        else:
            # Chunk the text before adding to collection
            chunks = self._chunking_manager.chunk_text(text)
            if not chunks:  # If no chunks were created, use the original text
                chunks = [text]

            # Create unique IDs for each chunk
            chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            chunk_metadatas = [
                {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunking_method": self._chunking_manager.get_chunking_method(),
                }
                for i in range(len(chunks))
            ]
        existing = collection.get(where={"url": url})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
        collection.upsert(documents=chunks, ids=chunk_ids, metadatas=chunk_metadatas)

    def bulk_add_documents(self, documents: list[tuple[str, str]]) -> None:
        with ThreadPoolExecutor() as executor:
            executor.map(lambda doc: self.add_document(*doc), documents)

    def clear_cache(self) -> None:
        """Clear the ChromaDB cache."""
        self._collections.clear()

    def query(
        self, collection_name: str, query_text: str, n_results: int = 5
    ) -> list[QueryResult]:
        """
        Query a specific collection and return results with their URLs.
        """
        collection = self.get_collection(collection_name)
        total_docs = collection.count()

        if total_docs == 0:
            return []
        # Dynamic initial retrieval logic
        if total_docs <= 100:
            # Small collection: Retrieve all documents for reranking
            top_k_initial = total_docs
        else:
            # Larger collection: Use percentage-based retrieval
            config = load_config(CONFIG_FILE)
            retrieval_config = config["embeddings"]["retrieval"]
            top_k_initial = min(
                max(
                    int(total_docs * retrieval_config["initial_percent"]),
                    retrieval_config["min_initial"],
                ),
                retrieval_config["max_initial"],
            )

        top_k_initial = max(1, top_k_initial)  # Prevent 0 or negative values
        # Step 1: Initial broad retrieval

        results = collection.query(
            query_texts=[query_text],
            n_results=top_k_initial,
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            logger.warning(f"No documents found for query: {query_text}")
            return []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        formatted_results = []
        # Step 2: Rerank with cross-encoder
        reranker = Reranker()
        top_indices = reranker.rerank(query_text, documents, top_k=n_results)
        logger.info(f"Reranked {top_k_initial} documents. Top indices: {top_indices}")
        # 3. Add validation for result indices
        formatted_results = []
        for idx in top_indices:
            try:
                if idx >= len(documents):
                    logger.error(
                        f"Invalid index {idx} for documents length {len(documents)}"
                    )
                    continue
                # Safely access metadata with fallbacks
                metadata = metadatas[idx] if idx < len(metadatas) else {}
                url = metadata.get("url", "No URL found")
                # Step 3: Format final results

                formatted_results.append(
                    {
                        "text": documents[idx],
                        "url": url,
                        "distance": distances[idx],
                        "metadata": metadata,
                    }
                )
                logger.info(f"Formatted result {formatted_results[-1]}")
            except Exception as e:
                logger.error(f"Error formatting result {idx}: {str(e)}")
                continue
        return formatted_results

    @classmethod
    def get_available_collections(cls) -> list[str]:
        """Get list of all available collections (websites)."""
        if cls._client is None:
            cls._client = chromadb.PersistentClient(
                path=cls.get_db_path(), settings=Settings(anonymized_telemetry=False)
            )

        return cls._client.list_collections()

    @classmethod
    def delete_collection(cls, collection_name: str) -> bool:
        """Delete a collection by name. Returns True if successful."""
        try:
            if cls._client is None:
                cls._client = chromadb.PersistentClient(
                    path=cls.get_db_path(),
                    settings=Settings(anonymized_telemetry=False),
                )

            # Delete from client
            cls._client.delete_collection(collection_name)

            # Remove from cache if exists
            if collection_name in cls._collections:
                del cls._collections[collection_name]

            return True
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {str(e)}")
            return False

    def update_document_metadata(
        self, collection_name: str, doc_id: str, metadata_update: dict
    ):
        """Update metadata for a specific document in a collection."""
        if not isinstance(metadata_update, dict):
            raise TypeError("metadata_update must be a dictionary")
        if any(k.startswith("_") for k in metadata_update.keys()):
            raise ValueError("metadata_update keys cannot start with '_'")

        collection = self.get_collection(collection_name)

        # Get current metadata
        results = collection.get(ids=[doc_id])
        if not results or not results["metadatas"]:
            return False

        # Merge existing metadata with updates
        current_metadata = results["metadatas"][0]
        updated_metadata = {**current_metadata, **metadata_update}

        # Update the document with new metadata
        collection.update(ids=[doc_id], metadatas=[updated_metadata])
        return True

    def get_all_documents(self, collection_name: str) -> list[dict]:
        """Get all documents from a collection with their IDs and metadata."""
        collection = self.get_collection(collection_name)
        return collection.get()

    def has_summaries(self, collection_name: str) -> bool:
        """Check if a collection has any documents with summaries."""
        try:
            collection = self.get_collection(collection_name)
            logger.debug(f"\nChecking summaries for {collection_name}")

            results = collection.get()
            if not results or not results["metadatas"]:
                logger.debug(f"No documents found in {collection_name}")
                return False

            logger.debug(
                f"Found {len(results['metadatas'])} documents in {collection_name}"
            )
            has_summary = False
            for i, metadata in enumerate(results["metadatas"]):
                logger.debug(f"Document {i} metadata: {metadata}")
                if metadata.get("summary"):
                    has_summary = True
                    break
            logger.debug(f"Collection {collection_name} has_summary: {has_summary}")
            return has_summary

        except Exception as e:
            logger.error(f"Error checking summaries for {collection_name}: {str(e)}")
            return False

    def has_matching_content(self, url: str, content: str) -> bool:
        """
        Check if a URL exists in the database and has matching content.

        Args:
            url: The URL to check
            content: The content to compare against

        Returns:
            bool: True if URL exists and content matches, False otherwise
        """
        try:
            collection_name = self.get_collection_name(url)
            collection = self.get_collection(collection_name)

            # Generate hash for the new content
            new_content_hash = blake2b(content.encode(), digest_size=16).hexdigest()

            # Get document by URL from metadata
            results = collection.get(where={"url": url}, include=["metadatas"])

            if not results["ids"]:
                return False

            # Check if content hash matches
            existing_hash = results["metadatas"][0].get("content_hash")
            return existing_hash == new_content_hash

        except Exception as e:
            logger.debug(f"Error checking content match: {e}")
            return False

    @classmethod
    def get_collection_name(cls, url: str = "") -> str:
        """Convert URL to a valid slug-based collection name with improved handling."""
        parsed = urlparse(url)
        base_name = None

        # Maintain original empty URL handling
        if not url.strip():
            return "default_collection"

        # Preserve original GitHub handling logic
        if "github.com" in parsed.netloc:
            path_parts = [p for p in parsed.path.split("/") if p]
            if len(path_parts) >= 2:
                # Keep original repository name extraction
                base_name = path_parts[1]

        # Maintain original domain handling behavior
        if not base_name:
            collection_name = (
                parsed.netloc.replace(".", "_").replace("-", "_").replace(":", "_")
            )
            if not collection_name:  # Preserve non-URL fallback
                collection_name = (
                    url.replace(".", "_").replace("-", "_").replace(":", "_")
                )
        else:
            collection_name = base_name

        # Keep original prefix logic
        if not collection_name[0].isalpha():
            collection_name = "collection_" + collection_name

        # Add normalization to prevent breaking changes
        return collection_name.strip("_").lower()[:63]
