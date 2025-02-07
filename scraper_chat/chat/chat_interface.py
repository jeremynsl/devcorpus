from typing import Union, Dict, List, Optional
from dotenv import load_dotenv
from scraper_chat.database.chroma_handler import ChromaHandler
import logging
from scraper_chat.core.llm_config import LLMConfig
from scraper_chat.plan_mode.plan_mode import PlanModeExecutor
from scraper_chat.config import load_config, CONFIG_FILE
from typing import Any, AsyncGenerator, Tuple
import re

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)


def format_context(results: List[Dict[str, Any]]) -> str:
    """Format search results into context for the LLM."""
    if not results:
        return "No relevant documentation found."

    context_parts = []
    for i, result in enumerate(results, 1):
        # Truncate text to a reasonable length while keeping it coherent
        text = result.get("text")
        if len(text) > 2000:  # Limit each context chunk
            text = text[:2000] + "..."

        # Build context with metadata
        context = [f"[{i}] Excerpt from {result.get('url', 'Unknown URL')}"]
        context.append(f"Relevance Score: {1 - result.get('distance', 1.0):.2f}")

        # Add summary if available
        metadata = result.get("metadata", {})
        if metadata.get("summary"):
            context.append(f"Summary: {metadata['summary']}")

        context.append(f"Content: {text}")
        context_parts.append("\n".join(context))

    return "\n---\n".join(context_parts)


def get_chat_prompt(query: str, context: str) -> str:
    """Create the RAG prompt for the LLM."""
    config = load_config(CONFIG_FILE)
    rag_prompt = config["chat"].get(
        "rag_prompt", "Context: {context}\nQuery: {query}\nResponse:"
    )
    return rag_prompt.replace("{context}", context).replace("{query}", query)


class ChatInterface:
    _config = load_config(CONFIG_FILE)

    def __init__(
        self, collection_names: Union[str, List[str]], model: Optional[str] = None
    ) -> None:
        """Initialize chat interface with ChromaDB collection(s)."""
        self.db = ChromaHandler()  # Initialize without collection
        self.collection_names = (
            [collection_names]
            if isinstance(collection_names, str)
            else collection_names
        )
        self.llm = LLMConfig(
            model or self._config["chat"]["models"]["default"]
        )  # Use provided model or default from config
        self.message_history = []  # Store message history
        self.max_history = self._config["chat"][
            "message_history_size"
        ]  # Get from config
        logger.info(
            f"Using LLM model: {model or self._config['chat']['models']['default']}"
        )

    def _add_to_history(self, role: str, content: str) -> None:
        """Add message to history and maintain max size"""
        message = {
            "role": role,
            "content": content,  # Keep content as plain string
            "cache_control": {"type": "ephemeral"},  # Add cache control at top level
        }
        self.message_history.append(message)

        # Keep history within size limit
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history :]

    async def get_response(
        self,
        query: str,
        n_results: int = 5,
    ) -> AsyncGenerator[Tuple[str, List[Dict[str, Any]]], None]:
        """
        Process user query across all collections and stream responses.
        Yields tuples of (chunk, excerpts) where chunk is a piece of the response
        and excerpts are the relevant context documents.
        n_results specifies the number of results per collection.
        """
        if not self.collection_names:
            yield ("Please select at least one documentation source to search.", [])
            return

        all_results = []

        # Search each collection
        for collection_name in self.collection_names:
            try:
                results = (
                    self.db.query(collection_name, query, n_results=n_results) or []
                )
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error querying collection {collection_name}: {str(e)}")
                continue

        if not all_results:
            yield ("No relevant information found in the selected documentation.", [])
            return

        # Sort all results by relevance score (1 - distance)
        all_results.sort(key=lambda x: x.get("distance", 1.0))

        # Format context and get response
        context = format_context(all_results)
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
                    yield (content, all_results)

            # Add complete response to history
            self._add_to_history("user", query)
            self._add_to_history("assistant", full_response)

        except Exception as e:
            logger.error(f"LLM Error: {str(e)}")
            yield (f"Error getting response from LLM: {str(e)}", all_results)

    def get_step_history(self, chat_history: list) -> list:
        """Parses raw chat messages into structured step history"""
        step_history = []
        current_step = None

        for msg in chat_history:
            if msg["role"] == "assistant":
                content = msg["content"]

                # Example message format:
                # "**Step 1**: Create project\n🔍 Query...\n💡 Solution..."
                if "**Step " in content:
                    # More robust parsing
                    try:
                        # Extract step number using regex

                        match = re.search(
                            r"\*\*Step\s*(\d+)\*\*:\s*(.+?)(?:\n|$)", content
                        )
                        if match:
                            step_num = int(match.group(1))
                            description = match.group(2).strip()

                            current_step = {
                                "step_number": step_num,
                                "description": description,
                                "query": None,
                                "solution": None,
                            }
                            step_history.append(current_step)

                            # Extract query and solution
                            query_match = re.search(
                                r"🔍\s*(.+?)(?:\n💡|$)", content, re.DOTALL
                            )
                            if query_match:
                                current_step["query"] = query_match.group(1).strip()

                            solution_match = re.search(
                                r"💡\s*(.+)$", content, re.DOTALL
                            )
                            if solution_match:
                                current_step["solution"] = solution_match.group(
                                    1
                                ).strip()

                    except Exception as e:
                        logger.error(f"Error parsing step history: {e}")
                        continue

        return step_history

    async def plan_mode_chat(
        self, message: str, history: list, collections: list, model: str
    ) -> AsyncGenerator[Tuple[List[dict], str], None]:
        """
        Handle Plan Mode:
          1) LLM outlines a plan
          2) Iteratively retrieve docs and generate solutions for each plan step
          3) Stream partial outputs
        """

        step_history = self.get_step_history(history)

        # Explicitly handle empty message
        if not message or message.strip() == "":
            # Yield a specific message for empty input
            history.append(
                {
                    "role": "assistant",
                    "content": "Please enter a message to start planning.",
                }
            )
            yield history, "Please enter a message to start planning."
            return

        # Initialize PlanModeExecutor
        plan_executor = PlanModeExecutor(collections, model)

        # Add user message and placeholder assistant message
        history.append({"role": "user", "content": message})

        try:
            # Create empty assistant message that we'll build incrementally
            partial_response = ""

            async for chunk, _ in plan_executor.plan_and_execute(message, step_history):
                # Append to partial response
                partial_response += chunk

                # Update or create assistant message
                if history and history[-1]["role"] == "assistant":
                    history[-1]["content"] = partial_response
                else:
                    history.append({"role": "assistant", "content": partial_response})

                # Yield the updated history and the chunk
                yield history, chunk

        except Exception as e:
            # Handle any errors during plan execution
            error_message = f"Error in plan mode: {str(e)}"
            history.append({"role": "assistant", "content": error_message})
            yield history, error_message
