import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from llm_config import LLMConfig

# Mock response data
MOCK_RESPONSE = MagicMock(
    choices=[MagicMock(message=MagicMock(content="Test response"))],
)

MOCK_STREAM_RESPONSE = [
    MagicMock(choices=[MagicMock(delta=MagicMock(content="Test "))]),
    MagicMock(choices=[MagicMock(delta=MagicMock(content="stream "))]),
    MagicMock(choices=[MagicMock(delta=MagicMock(content="response"))])
]

@pytest.fixture
def mock_config():
    return {
        "chat": {
            "system_prompt": "Test system prompt",
            "models": {
                "available": ["test-model"],
                "default": "test-model"
            }
        }
    }

@pytest.fixture
def llm_config(mock_config):
    with patch("llm_config.config", mock_config):
        yield LLMConfig("test-model")

@pytest.mark.asyncio
async def test_get_response_no_stream(llm_config):
    """Test getting a non-streaming response"""
    with patch("litellm.acompletion", AsyncMock(return_value=MOCK_RESPONSE)):
        response = await llm_config.get_response([{"role": "user", "content": "test"}], stream=False)
        assert response == "Test response"

@pytest.mark.asyncio
async def test_get_response_stream(llm_config):
    """Test getting a streaming response"""
    mock_acompletion = AsyncMock()
    mock_acompletion.return_value.__aiter__.return_value = MOCK_STREAM_RESPONSE
    
    with patch("litellm.acompletion", mock_acompletion):
        response = await llm_config.get_response([{"role": "user", "content": "test"}], stream=True)
        chunks = []
        async for chunk in response:
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        assert "".join(chunks) == "Test stream response"

@pytest.mark.asyncio
async def test_get_response_empty_choices(llm_config):
    """Test handling empty choices in response"""
    empty_response = MagicMock(choices=[])
    with patch("litellm.acompletion", AsyncMock(return_value=empty_response)):
        response = await llm_config.get_response([{"role": "user", "content": "test"}], stream=False)
        assert response == ""

@pytest.mark.asyncio
async def test_system_prompt_included(llm_config):
    """Test that system prompt is included in messages"""
    mock_acompletion = AsyncMock(return_value=MOCK_RESPONSE)
    
    with patch("litellm.acompletion", mock_acompletion):
        await llm_config.get_response([{"role": "user", "content": "test"}], stream=False)
        
        # Check that system prompt was included
        calls = mock_acompletion.call_args_list
        assert len(calls) == 1
        messages = calls[0].kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Test system prompt"
