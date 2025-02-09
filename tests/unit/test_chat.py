import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent.parent))
from scraper_chat.chat.chat_interface import (
    ChatInterface,
    format_context,
    get_chat_prompt,
)
from scraper_chat.database.chroma_handler import ChromaHandler


@pytest.fixture
def mock_config():
    return {
        "chat": {
            "message_history_size": 2,
            "system_prompt": "Test system prompt",
            "models": {"available": ["test-model"], "default": "test-model"},
        }
    }


@pytest.fixture
def mock_chroma():
    handler = MagicMock()
    handler.query.return_value = (
        [{"text": "Test document", "url": "http://test.com", "distance": 0.5}],
        15.0,
    )
    return handler


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    # Mock streaming response
    mock_response = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content="Test "))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content="response"))]),
    ]
    llm.get_response.return_value.__aiter__.return_value = mock_response
    return llm


@pytest.fixture
def chat_interface(mock_config, mock_chroma, mock_llm):
    with (
        patch("scraper_chat.config.config", mock_config),
        patch(
            "scraper_chat.chat.chat_interface.ChromaHandler", return_value=mock_chroma
        ),
        patch("scraper_chat.chat.chat_interface.LLMConfig", return_value=mock_llm),
    ):
        return ChatInterface(["test_collection"])


def test_format_context():
    """Test context formatting"""
    results = [{"text": "Test document", "url": "http://test.com", "distance": 0.5}]
    context = format_context(results)
    assert "[1]" in context
    assert "Test document" in context
    assert "http://test.com" in context

    # Test empty results
    assert "No relevant documentation found" in format_context([])


def test_get_chat_prompt():
    """Test prompt formatting"""
    prompt = get_chat_prompt("test query", "test context", avg_score=15.0)
    assert "{context}" not in prompt
    assert "{query}" not in prompt
    assert "test context" in prompt
    assert "test query" in prompt


def test_get_chat_prompt_config_loading():
    """Test loading RAG prompt from config"""
    with (
        patch("builtins.open", create=True) as mock_open,
        patch("json.load") as mock_load,
    ):
        # Mock config with specific RAG prompt
        mock_load.return_value = {
            "chat": {
                "rag_prompt": "Context: {context}\nQuery: {query}\nResponse:",
                "rag_prompt_high_quality": "High quality prompt",
                "rag_prompt_low_quality": "Low quality prompt",
            }
        }

        # Test high quality prompt
        prompt = get_chat_prompt("test query", "test context", avg_score=15.0)
        assert "High quality prompt" in prompt

        # Test low quality prompt
        prompt = get_chat_prompt("test query", "test context", avg_score=5.0)
        assert "Low quality prompt" in prompt


@pytest.mark.asyncio
async def test_get_response_no_collections(chat_interface):
    """Test response when no collections selected"""
    chat_interface.collection_names = []
    async for response, excerpts in chat_interface.get_response("test"):
        assert "Please select" in response
        assert excerpts == []
        break


@pytest.mark.asyncio
async def test_get_response_no_results(chat_interface, mock_chroma):
    """Test response when no results found"""
    mock_chroma.query.return_value = ([], 15.0)
    async for response, excerpts in chat_interface.get_response("test"):
        assert "No relevant information" in response
        assert excerpts == []
        break


@pytest.mark.asyncio
async def test_get_response_with_excerpts(chat_interface):
    """Test response with excerpts"""
    # Mock the query method to return some results
    with patch.object(ChromaHandler, "query") as mock_query:
        mock_query.return_value = (
            [{"text": "test excerpt", "url": "http://test.com"}],
            15.0,
        )

        async for response, excerpts in chat_interface.get_response(
            "test",
        ):
            assert isinstance(response, str)
            assert isinstance(excerpts, list)
            assert len(excerpts) > 0


@pytest.mark.asyncio
async def test_message_history_limit(chat_interface):
    """Test message history size limit"""
    # Add messages up to limit
    for i in range(3):  # More than message_history_size
        async for _ in chat_interface.get_response(f"test {i}"):
            pass

    # Check that history is limited
    assert len(chat_interface.message_history) <= chat_interface.max_history


@pytest.mark.asyncio
async def test_error_handling(chat_interface, mock_llm):
    """Test error handling in get_response"""
    # Mock query to return valid results
    with patch.object(ChromaHandler, "query") as mock_query:
        mock_query.return_value = (
            [{"text": "test excerpt", "url": "http://test.com"}],
            15.0,
        )

        # Make LLM raise an error
        mock_llm.get_response.side_effect = Exception("Test error")
        async for response, excerpts in chat_interface.get_response("test"):
            assert "Error" in response
            assert isinstance(excerpts, list)
            break


def test_format_context_long_text():
    """Test context formatting with long text"""
    results = [
        {
            "text": "A" * 3000,  # Longer than 2000 characters
            "url": "http://longtext.com",
            "distance": 0.3,
            "metadata": {"summary": "Long text summary"},
        }
    ]
    context = format_context(results)

    # Verify truncation
    assert len(context) <= 2500  # Allowing some margin for additional formatting
    assert context.endswith("...")

    # Verify metadata inclusion
    assert "Long text summary" in context
    assert "http://longtext.com" in context


def test_chat_interface_initialization():
    """Test ChatInterface initialization with various inputs"""
    # Test with single collection name (string)
    mock_chroma = MagicMock()
    mock_llm = MagicMock()

    with (
        patch(
            "scraper_chat.database.chroma_handler.ChromaHandler._instance", mock_chroma
        ),
        patch("scraper_chat.core.llm_config.LLMConfig", return_value=mock_llm),
        patch(
            "scraper_chat.config.config",
            return_value={"chat": {"models": {"default": "test-model"}}},
        ),
    ):
        chat_interface = ChatInterface("test_collection")
        assert isinstance(chat_interface.db, MagicMock)
        assert len(chat_interface.message_history) == 0

    # Test with multiple collection names (list)
    with (
        patch(
            "scraper_chat.database.chroma_handler.ChromaHandler._instance", mock_chroma
        ),
        patch("scraper_chat.core.llm_config.LLMConfig", return_value=mock_llm),
        patch(
            "scraper_chat.config.config",
            return_value={"chat": {"models": {"default": "test-model"}}},
        ),
    ):
        chat_interface = ChatInterface(["collection1", "collection2"])
        assert isinstance(chat_interface.db, MagicMock)


def test_add_to_history():
    """Test _add_to_history method"""
    mock_chroma = MagicMock()
    mock_llm = MagicMock()

    with (
        patch(
            "scraper_chat.database.chroma_handler.ChromaHandler._instance", mock_chroma
        ),
        patch("scraper_chat.core.llm_config.LLMConfig", return_value=mock_llm),
        patch(
            "scraper_chat.config.config",
            return_value={
                "chat": {"models": {"default": "test-model"}, "message_history_size": 3}
            },
        ),
    ):
        chat_interface = ChatInterface("test_collection")

        # Manually set max_history to avoid MagicMock issues
        chat_interface.max_history = 3

        # Add messages
        chat_interface._add_to_history("user", "First message")
        chat_interface._add_to_history("assistant", "First response")
        chat_interface._add_to_history("user", "Second message")
        chat_interface._add_to_history("assistant", "Second response")
        chat_interface._add_to_history("user", "Third message")

        # Verify history size limit
        assert len(chat_interface.message_history) == 3

        # Verify most recent messages are kept
        assert chat_interface.message_history[0]["content"] == "Second message"
        assert chat_interface.message_history[1]["content"] == "Second response"
        assert chat_interface.message_history[2]["content"] == "Third message"


def test_get_step_history():
    """Test get_step_history method parsing"""
    chat_interface = ChatInterface("test_collection")

    # Simulate chat history with plan mode messages
    chat_history = [
        {"role": "user", "content": "Create a web app"},
        {
            "role": "assistant",
            "content": "**Step 1**: Design UI\n🔍 Query for UI design\n💡 Solution for UI design",
        },
        {
            "role": "assistant",
            "content": "**Step 2**: Implement backend\n🔍 Query for backend\n💡 Solution for backend",
        },
    ]

    step_history = chat_interface.get_step_history(chat_history)

    # Verify step history parsing
    assert len(step_history) == 2
    assert step_history[0]["step_number"] == 1
    assert step_history[0]["description"] == "Design UI"
    assert step_history[0]["query"] == "Query for UI design"
    assert step_history[0]["solution"] == "Solution for UI design"

    assert step_history[1]["step_number"] == 2
    assert step_history[1]["description"] == "Implement backend"
    assert step_history[1]["query"] == "Query for backend"
    assert step_history[1]["solution"] == "Solution for backend"


@pytest.mark.asyncio
async def test_plan_mode_chat_empty_message():
    """Test plan_mode_chat with empty message"""
    mock_chroma = MagicMock()
    mock_llm = MagicMock()
    mock_plan = MagicMock()

    with (
        patch(
            "scraper_chat.database.chroma_handler.ChromaHandler._instance", mock_chroma
        ),
        patch("scraper_chat.core.llm_config.LLMConfig", return_value=mock_llm),
        patch(
            "scraper_chat.plan_mode.plan_mode.PlanModeExecutor", return_value=mock_plan
        ),
        patch(
            "scraper_chat.config.config",
            return_value={"chat": {"models": {"default": "test-model"}}},
        ),
    ):
        chat_interface = ChatInterface("test_collection")

        # Simulate empty message
        history = []
        async for updated_history, note in chat_interface.plan_mode_chat(
            "", history, ["test_collection"], "test-model"
        ):
            # Check for the specific note about empty message
            assert "Please enter a message" in note
            assert len(updated_history) > 0
            assert updated_history[-1]["role"] == "assistant"
            assert "Please enter a message" in updated_history[-1]["content"]
            break
