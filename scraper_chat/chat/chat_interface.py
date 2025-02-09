"""
Interface for RAG-based chat functionality using ChromaDB and LLMs.
Provides streaming responses with relevant context from stored documentation.
Supports both regular chat and plan mode for complex queries.
"""

from typing import Union, Dict, List, Optional
from dotenv import load_dotenv
from scraper_chat.database.chroma_handler import ChromaHandler
import logging
from scraper_chat.core.llm_config import LLMConfig
from scraper_chat.plan_mode.plan_mode import PlanModeExecutor
from scraper_chat.config import load_config, CONFIG_FILE
from typing import Any, AsyncGenerator, Tuple
import re

load_dotenv()
logger = logging.getLogger(__name__)


def format_context(results: List[Dict[str, Any]]) -> str:
    """
    Format search results into a context string for the LLM.

    Args:
        results: List of search results from ChromaDB, each containing text and metadata

    Returns:
        Formatted context string with numbered excerpts, URLs, and relevance scores
    """
    if not results:
        return "No relevant documentation found."

    context_parts = []
    for i, result in enumerate(results, 1):
        text = result.get("text")
        if len(text) > 2000:
            text = text[:2000] + "..."

        context = [f"[{i}] Excerpt from {result.get('url', 'Unknown URL')}"]
        context.append(f"Relevance Score: {1 - result.get('distance', 1.0):.2f}")

        metadata = result.get("metadata", {})
        if metadata.get("summary"):
            context.append(f"Summary: {metadata['summary']}")

        context.append(f"Content: {text}")
        context_parts.append("\n".join(context))

    return "\n---\n".join(context_parts)


def get_chat_prompt(query: str, context: str, avg_score: float) -> str:
    """
    Create the RAG prompt by combining user query and context.

    Args:
        query: User's question or request
        context: Formatted context from relevant documents
        avg_score: Average reranking score of retrieved documents

    Returns:
        Complete prompt string for the LLM
    """
    config = load_config(CONFIG_FILE)
    chat_config = config["chat"]

    # Use appropriate prompt based on retrieval quality
    if avg_score >= 10:
        prompt_key = "rag_prompt_high_quality"
        logger.info(f"Using high quality prompt (avg_score={avg_score})")
    else:
        prompt_key = "rag_prompt_low_quality"
        logger.info(f"Using low quality prompt (avg_score={avg_score})")

    # Get prompt, falling back to default if quality-specific one not found
    rag_prompt = chat_config.get(prompt_key)
    if not rag_prompt:
        logger.warning(f"Prompt {prompt_key} not found, using default prompt")
        rag_prompt = chat_config["rag_prompt"]

    return rag_prompt.replace("{context}", context).replace("{query}", query)


class ChatInterface:
    """
    Interface for RAG-based chat functionality.
    Handles document retrieval, LLM interaction, and response streaming.
    """

    _config = load_config(CONFIG_FILE)

    def __init__(
        self, collection_names: Union[str, List[str]], model: Optional[str] = None
    ) -> None:
        """
        Initialize chat interface.

        Args:
            collection_names: Name(s) of ChromaDB collections to search
            model: Optional LLM model name, uses default from config if not specified
        """
        self.db = ChromaHandler()
        self.collection_names = (
            [collection_names]
            if isinstance(collection_names, str)
            else collection_names
        )
        self.llm = LLMConfig(model or self._config["chat"]["models"]["default"])
        self.message_history = []
        self.max_history = self._config["chat"]["message_history_size"]
        logger.info(
            f"Using LLM model: {model or self._config['chat']['models']['default']}"
        )

    def _add_to_history(self, role: str, content: str) -> None:
        """
        Add message to chat history and maintain maximum history size.

        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        message = {
            "role": role,
            "content": content,
            "cache_control": {"type": "ephemeral"},
        }
        self.message_history.append(message)

        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history :]

    async def get_response(
        self,
        query: str,
        n_results: int = 5,
    ) -> AsyncGenerator[Tuple[str, List[Dict[str, Any]]], None]:
        """
        Process query and stream LLM responses with relevant context.

        Args:
            query: User's question or request
            n_results: Number of results to retrieve per collection

        Yields:
            Tuples of (response_chunk, context_documents)
        """
        if not self.collection_names:
            yield ("Please select at least one documentation source to search.", [])
            return

        all_results = []
        total_score = 0
        result_count = 0

        for collection_name in self.collection_names:
            try:
                results, avg_score = self.db.query(
                    collection_name, query, n_results=n_results
                )
                if results:
                    all_results.extend(results)
                    total_score += avg_score * len(results)
                    result_count += len(results)
            except Exception as e:
                logger.error(f"Error querying collection {collection_name}: {str(e)}")
                continue

        if not all_results:
            yield ("No relevant information found in the selected documentation.", [])
            return

        # Calculate overall average score across all collections
        avg_score = total_score / result_count if result_count > 0 else 0
        logger.info(f"Overall average reranking score: {avg_score}")

        all_results.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        context = format_context(all_results)
        prompt = get_chat_prompt(query, context, avg_score)

        try:
            response_stream = await self.llm.get_response(prompt, stream=True)
            full_response = ""
            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield (content, all_results)

            self._add_to_history("user", query)
            self._add_to_history("assistant", full_response)

        except Exception as e:
            logger.error(f"LLM Error: {str(e)}")
            yield (f"Error getting response from LLM: {str(e)}", all_results)

    def get_step_history(self, chat_history: list) -> list:
        """
        Parse chat messages into structured step history for plan mode.

        Args:
            chat_history: List of chat messages

        Returns:
            List of structured steps with number, description, query, and solution
        """
        step_history = []
        current_step = None

        for msg in chat_history:
            if msg["role"] == "assistant":
                content = msg["content"]
                if "**Step " in content:
                    try:
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
        Execute chat in plan mode, breaking complex queries into steps.

        Args:
            message: User's request
            history: Chat history
            collections: List of collections to search
            model: LLM model to use

        Yields:
            Tuples of (updated_history, response_chunk)
        """
        step_history = self.get_step_history(history)

        if not message or message.strip() == "":
            history.append(
                {
                    "role": "assistant",
                    "content": "Please enter a message to start planning.",
                }
            )
            yield history, "Please enter a message to start planning."
            return

        plan_executor = PlanModeExecutor(collections, model)
        history.append({"role": "user", "content": message})

        try:
            partial_response = ""
            async for chunk, _ in plan_executor.plan_and_execute(message, step_history):
                partial_response += chunk

                if history and history[-1]["role"] == "assistant":
                    history[-1]["content"] = partial_response
                else:
                    history.append({"role": "assistant", "content": partial_response})

                yield history, chunk

        except Exception as e:
            error_message = f"Error in plan mode: {str(e)}"
            history.append({"role": "assistant", "content": error_message})
            yield history, error_message
