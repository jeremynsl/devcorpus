import chromadb
from chromadb.config import Settings
import logging
from urllib.parse import urlparse
from embeddings import EmbeddingManager
from chunking import ChunkingManager

logger = logging.getLogger("ChromaDB")
logger.setLevel(logging.DEBUG)

# Add a console handler if none exists
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Default DB path, can be overridden for testing
DEFAULT_DB_PATH = "docs_database.db"

class ChromaHandler:
    """Handler for ChromaDB operations with singleton pattern"""
    _instance = None
    _client = None
    _collections = {}
    _embedding_manager = EmbeddingManager()
    _chunking_manager = ChunkingManager()
    _db_path = None
    
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
    
    def __new__(cls, collection_name: str = None):
        """Singleton pattern to ensure one database connection."""
        if cls._instance is None:
            cls._instance = super(ChromaHandler, cls).__new__(cls)
            cls._client = chromadb.Client(Settings(
                persist_directory=cls.get_db_path(),
                is_persistent=True
            ))
        return cls._instance
    
    @classmethod
    def get_collection_name(cls, url: str) -> str:
        """Convert URL to a valid collection name."""
        parsed = urlparse(url)
        # Use domain name as collection name, removing special characters
        collection_name = parsed.netloc.replace(".", "_").replace("-", "_").replace(":", "_")
        
        # For non-URL inputs, use the raw input
        if not collection_name:
            if not url:
                return "default_collection"
            collection_name = url.replace(".", "_").replace("-", "_").replace(":", "_")
            
        # Add prefix to ensure it starts with a letter
        if not collection_name[0].isalpha():
            collection_name = "collection_" + collection_name
        return collection_name
    
    def __init__(self, collection_name: str = None):
        """
        Initialize ChromaDB collection.
        """
        if collection_name and collection_name not in self._collections:
            self._collections[collection_name] = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._embedding_manager.embedding_function,
                metadata={"description": "Scraped website content"}
            )
    
    def get_collection(self, collection_name: str):
        """Get or create a collection by name."""
        if collection_name not in self._collections:
            self._collections[collection_name] = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._embedding_manager.embedding_function
            )
        return self._collections[collection_name]
        
    def add_document(self, text: str, url: str):
        """
        Add a document to the collection with its URL as the ID.
        Handles duplicate URLs by using upsert.
        """
        if not text.strip():
            return
            
        # Get collection name from URL
        collection_name = self.get_collection_name(url)
        collection = self.get_collection(collection_name)
        
        # Use URL as ID but make it safe for ChromaDB
        doc_id = url.replace("/", "_").replace(":", "_")
        
        # Check for existing metadata
        existing_metadata = {}
        try:
            results = collection.get(ids=[doc_id])
            if results and results['metadatas']:
                existing_metadata = results['metadatas'][0]
        except:
            pass
            
        # Merge with new metadata, preserving existing fields
        metadata = {**existing_metadata, "url": url}
        
        # Chunk the text before adding to collection
        chunks = self._chunking_manager.chunk_text(text)
        if not chunks:  # If no chunks were created, use the original text
            chunks = [text]
            
        # Create unique IDs for each chunk
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        chunk_metadatas = [{
            **metadata,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "chunking_method": self._chunking_manager.get_chunking_method()
        } for i in range(len(chunks))]
        
        collection.upsert(
            documents=chunks,
            ids=chunk_ids,
            metadatas=chunk_metadatas
        )
        
    def query(self, collection_name: str, query_text: str, n_results: int = 5) -> list[dict]:
        """
        Query a specific collection and return results with their URLs.
        """
        collection = self.get_collection(collection_name)
        
        # Get total count of documents in collection
        count = collection.count()
        if count == 0:
            return []
            
        # Adjust n_results if we have fewer documents than requested
        actual_n_results = min(n_results, count)
        
        results = collection.query(
            query_texts=[query_text],
            n_results=actual_n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        for idx, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            formatted_results.append({
                'text': doc,
                'url': metadata['url'],
                'distance': distance,
                'metadata': metadata  # Include full metadata
            })
            
        return formatted_results
        
    @classmethod
    def get_available_collections(cls) -> list[str]:
        """Get list of all available collections (websites)."""
        if cls._client is None:
            cls._client = chromadb.Client(Settings(
                persist_directory=cls.get_db_path(),
                is_persistent=True
            ))
        return cls._client.list_collections()

    @classmethod
    def delete_collection(cls, collection_name: str) -> bool:
        """Delete a collection by name. Returns True if successful."""
        try:
            if cls._client is None:
                cls._client = chromadb.Client(Settings(
                    persist_directory=cls.get_db_path(),
                    is_persistent=True
                ))
            
            # Delete from client
            cls._client.delete_collection(collection_name)
            
            # Remove from cache if exists
            if collection_name in cls._collections:
                del cls._collections[collection_name]
                
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {str(e)}")
            return False

    def update_document_metadata(self, collection_name: str, doc_id: str, metadata_update: dict):
        """Update metadata for a specific document in a collection."""
        collection = self.get_collection(collection_name)
        
        # Get current metadata
        results = collection.get(ids=[doc_id])
        if not results or not results['metadatas']:
            return False
            
        # Merge existing metadata with updates
        current_metadata = results['metadatas'][0]
        updated_metadata = {**current_metadata, **metadata_update}
        
        # Update the document with new metadata
        collection.update(
            ids=[doc_id],
            metadatas=[updated_metadata]
        )
        return True
        
    def get_all_documents(self, collection_name: str) -> list:
        """Get all documents from a collection with their IDs and metadata."""
        collection = self.get_collection(collection_name)
        return collection.get()

    def has_summaries(self, collection_name: str) -> bool:
        """Check if any documents in the collection have summaries."""
        collection = self.get_collection(collection_name)
        try:
            print(f"\nChecking summaries for {collection_name}")  # Debug print
            results = collection.get()
            if not results or not results['metadatas']:
                print(f"No documents found in {collection_name}")  # Debug print
                return False
                
            # Log metadata for debugging
            print(f"Found {len(results['metadatas'])} documents in {collection_name}")  # Debug print
            for i, metadata in enumerate(results['metadatas']):
                print(f"Document {i} metadata: {metadata}")  # Debug print
                
            # Check if any document has a summary
            has_summary = any(metadata.get('summary') for metadata in results['metadatas'])
            print(f"Collection {collection_name} has_summary: {has_summary}")  # Debug print
            return has_summary
            
        except Exception as e:
            print(f"Error checking summaries for {collection_name}: {str(e)}")  # Debug print
            logger.error(f"Error checking summaries for {collection_name}: {str(e)}")
            return False
