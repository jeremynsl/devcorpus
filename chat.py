import json
from typing import Union, Dict, List
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
            
        # Build context with metadata
        context = [f"[{i}] Excerpt from {result['url']}"]
        context.append(f"Relevance Score: {1 - result['distance']:.2f}")
        
        # Add summary if available
        if 'metadata' in result and result['metadata'].get('summary'):
            context.append(f"Summary: {result['metadata']['summary']}")
            
        context.append(f"Content: {text}")
        context_parts.append("\n".join(context))
    
    return "\n---\n".join(context_parts)

def get_chat_prompt(query: str, context: str) -> str:
    """Create the RAG prompt for the LLM."""
    with open("scraper_config.json", "r") as f:
        config = json.load(f)
    
    rag_prompt = config["chat"]["rag_prompt"]
    return rag_prompt.format(context=context, query=query)

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
        Process user query across all collections and stream responses.
        Yields tuples of (chunk, excerpts) where chunk is a piece of the response
        and excerpts are the relevant context documents.
        """
        if not self.collection_names:
            yield ("Please select at least one documentation source to search.", [])
            return
            
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
            yield ("No relevant information found in the selected documentation.", [])
            return
        
        # Sort results by distance (lower is better)
        all_results.sort(key=lambda x: x['distance'])
        
        # Take top N results across all collections
        top_results = all_results[:n_results]
        
        # Format context and get response
        context = format_context(top_results)
        prompt = get_chat_prompt(query, context)
        
        try:
            # Get streaming response
            response_stream = await self.llm.get_response(prompt, stream=True)
            
            # Stream chunks while building full response
            full_response = ""
            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield (content, top_results)
            
            # Add complete response to history
            self._add_to_history("user", query)
            self._add_to_history("assistant", full_response)
            
        except Exception as e:
            logger.error(f"LLM Error: {str(e)}")
            yield (f"Error getting response from LLM: {str(e)}", top_results)
        
    def run_chat_loop(self):
        """Run interactive chat loop."""
        print("\nChat Interface Ready (Ctrl+C to exit)")
        print("Enter your question:")
        
        while True:
            try:
                query = input("\nYou: ").strip()
                if not query:
                    continue
                    
                async def get_response_stream():
                    async for response, excerpts in self.get_response(query, return_excerpts=True):
                        print(f"\nAssistant: {response}")
                        if excerpts:
                            print("Excerpts:")
                            for excerpt in excerpts:
                                print(excerpt)
                
                import asyncio
                asyncio.run(get_response_stream())
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                continue
