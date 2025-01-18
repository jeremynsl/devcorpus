import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from chat import ChatInterface, format_context, get_chat_prompt

@pytest.fixture
def mock_config():
    return {
        "chat": {
            "message_history_size": 2,
            "system_prompt": "Test system prompt",
            "models": {
                "available": ["test-model"],
                "default": "test-model"
            }
        }
    }

@pytest.fixture
def mock_chroma():
    handler = MagicMock()
    handler.query.return_value = [
        {
            "text": "Test document",
            "url": "http://test.com",
            "distance": 0.5
        }
    ]
    return handler

@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    # Mock streaming response
    mock_response = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content="Test "))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content="response"))])
    ]
    llm.get_response.return_value.__aiter__.return_value = mock_response
    return llm

@pytest.fixture
def chat_interface(mock_config, mock_chroma, mock_llm):
    with patch("chat.config", mock_config), \
         patch("chat.ChromaHandler", return_value=mock_chroma), \
         patch("chat.LLMConfig", return_value=mock_llm):
        return ChatInterface(["test_collection"])

def test_format_context():
    """Test context formatting"""
    results = [
        {
            "text": "Test document",
            "url": "http://test.com",
            "distance": 0.5
        }
    ]
    context = format_context(results)
    assert "[1]" in context
    assert "Test document" in context
    assert "http://test.com" in context
    
    # Test empty results
    assert "No relevant documentation found" in format_context([])

def test_get_chat_prompt():
    """Test prompt formatting"""
    prompt = get_chat_prompt("test query", "test context")
    assert "test query" in prompt
    assert "test context" in prompt

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
    mock_chroma.query.return_value = []
    async for response, excerpts in chat_interface.get_response("test"):
        assert "No relevant information" in response
        assert excerpts == []
        break

@pytest.mark.asyncio
async def test_get_response_with_excerpts(chat_interface):
    """Test response with excerpts"""
    async for response, excerpts in chat_interface.get_response("test", return_excerpts=True):
        assert isinstance(response, str)
        assert isinstance(excerpts, list)
        assert len(excerpts) > 0
        break

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
    mock_llm.get_response.side_effect = Exception("Test error")
    async for response, excerpts in chat_interface.get_response("test"):
        assert "Error" in response
        assert isinstance(excerpts, list)
        break
