import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import logging
from urllib.parse import urlparse

logger = logging.getLogger("ChromaDB")
logger.setLevel(logging.DEBUG)

DB_PATH = "docs_database.db"

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="avsolatorio/GIST-Embedding-v0"
)

class ChromaHandler:
    _instance = None
    _client = None
    _collections = {}
    
    @classmethod
    def get_collection_name(cls, url: str) -> str:
        """Convert URL to a valid collection name."""
        parsed = urlparse(url)
        # Use domain name as collection name, removing special characters
        collection_name = parsed.netloc.replace(".", "_").replace("-", "_")
        # Ensure name meets ChromaDB requirements
        if not collection_name:
            collection_name = "default_collection"
        # Add prefix to ensure it starts with a letter
        if not collection_name[0].isalpha():
            collection_name = "collection_" + collection_name
        return collection_name
    
    def __new__(cls, collection_name: str = None):
        """Singleton pattern to ensure one database connection."""
        if cls._instance is None:
            cls._instance = super(ChromaHandler, cls).__new__(cls)
            # Initialize the client only once
            cls._client = chromadb.Client(Settings(
                persist_directory=DB_PATH,
                is_persistent=True
            ))
        return cls._instance
    
    def __init__(self, collection_name: str = None):
        """
        Initialize ChromaDB collection.
        """
        if collection_name and collection_name not in self._collections:
            self._collections[collection_name] = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=sentence_transformer_ef,
                metadata={"description": "Scraped website content"}
            )
    
    def get_collection(self, collection_name: str):
        """Get a specific collection."""
        if collection_name not in self._collections:
            self._collections[collection_name] = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=sentence_transformer_ef
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
        
        collection.upsert(
            documents=[text],
            ids=[doc_id],
            metadatas=[{"url": url}]
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
                'distance': distance
            })
            
        return formatted_results
        
    @classmethod
    def get_available_collections(cls) -> list[str]:
        """Get list of all available collections (websites)."""
        if cls._client is None:
            cls._client = chromadb.Client(Settings(
                persist_directory=DB_PATH,
                is_persistent=True
            ))
        return cls._client.list_collections()

    @classmethod
    def delete_collection(cls, collection_name: str) -> bool:
        """Delete a collection by name. Returns True if successful."""
        try:
            if cls._client is None:
                cls._client = chromadb.Client(Settings(
                    persist_directory=DB_PATH,
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
