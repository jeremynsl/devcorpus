import os
import json
from typing import List, Dict, Union
from dotenv import load_dotenv
from chroma import ChromaHandler
import logging
from llm_config import LLMConfig

# Load environment variables
load_dotenv()

# Load config
with open("scraper_config.json", "r") as f:
    config = json.load(f)

# Set up logging
logger = logging.getLogger("ChatInterface")
logger.setLevel(logging.DEBUG)

# Configure console handler if not already configured
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s | %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def format_context(results: List[Dict]) -> str:
    """Format search results into context for the LLM."""
    if not results:
        return "No relevant documentation found."
        
    context_parts = []
    for i, result in enumerate(results, 1):
        # Truncate text to a reasonable length while keeping it coherent
        text = result['text']
        if len(text) > 2000:  # Limit each context chunk
            text = text[:2000] + "..."
            
        context_parts.append(
            f"[{i}] Excerpt from {result['url']}\n"
            f"Relevance Score: {1 - result['distance']:.2f}\n"
            f"Content: {text}\n"
        )
    
    return "\n---\n".join(context_parts)

def get_chat_prompt(query: str, context: str) -> str:
    """Create the RAG prompt for the LLM."""
    return f"""You are a helpful expert, answering questions based on the provided context.
Use ONLY the following documentation excerpts to answer the question. If you cannot answer based on these excerpts, say so.
Always cite your sources using the [number] format when referencing information.

DOCUMENTATION EXCERPTS:
{context}

USER QUESTION: {query}

Please provide a clear and concise answer, citing specific sources with [number] format. If multiple sources support a point, cite all of them.
If you cannot answer the question based on the provided context, say so clearly."""

class ChatInterface:
    def __init__(self, collection_names: Union[str, List[str]], model: str = None):
        """Initialize chat interface with ChromaDB collection(s)."""
        self.db = ChromaHandler()  # Initialize without collection
        self.collection_names = [collection_names] if isinstance(collection_names, str) else collection_names
        self.llm = LLMConfig(model or config["chat"]["models"]["default"])  # Use provided model or default from config
        self.message_history = []  # Store message history
        self.max_history = config["chat"]["message_history_size"]  # Get from config
        logger.info(f"Using LLM model: {model or config['chat']['models']['default']}")

    def _add_to_history(self, role: str, content: str):
        """Add message to history and maintain max size"""
        message = {
            "role": role,
            "content": content,  # Keep content as plain string
            "cache_control": {"type": "ephemeral"}  # Add cache control at top level
        }
        self.message_history.append(message)
        
        # Keep history within size limit
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history:]
        
    async def get_response(self, query: str, n_results: int = 5, return_excerpts: bool = False):
        """
        Process user query across all selected collections:
        1. Search each ChromaDB collection for relevant context
        2. Format combined context and create prompt
        3. Get response from LLM with history
        """
        if not self.collection_names:
            response = "Please select at least one documentation source to search."
            return (response, []) if return_excerpts else response
            
        all_results = []
        
        # Search each collection
        for collection_name in self.collection_names:
            try:
                results = self.db.query(collection_name, query, n_results=n_results)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error querying collection {collection_name}: {str(e)}")
                continue
        
        if not all_results:
            response = "No relevant information found in the selected documentation."
            return (response, []) if return_excerpts else response
        
        # Sort results by distance (lower is better)
        all_results.sort(key=lambda x: x['distance'])
        
        # Take top N results across all collections
        top_results = all_results[:n_results]
        
        # Format context and get response
        context = format_context(top_results)
        prompt = get_chat_prompt(query, context)
        
        # Add user query to history
        self._add_to_history("user", prompt)
        
        # Get response from LLM using same instance for caching
        try:
            response = await self.llm.get_response(self.message_history)
            # Add assistant response to history
            self._add_to_history("assistant", response)
        except Exception as e:
            logger.error(f"LLM Error: {str(e)}")
            response = f"Error getting response from LLM: {str(e)}"
        
        if return_excerpts:
            return response, top_results
        return response
        
    def run_chat_loop(self):
        """Run interactive chat loop."""
        print("\nChat Interface Ready (Ctrl+C to exit)")
        print("Enter your question:")
        
        while True:
            try:
                query = input("\nYou: ").strip()
                if not query:
                    continue
                    
                response = self.get_response(query)
                print(f"\nAssistant: {response}")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                continue
