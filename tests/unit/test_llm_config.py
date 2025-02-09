import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import sys
from pathlib import Path
from tenacity import RetryError
import logging

# Add parent directory to path to import from scraper_chat
sys.path.append(str(Path(__file__).parent.parent.parent))

from scraper_chat.core.llm_config import LLMConfig
from litellm import (
    Timeout,
    BadRequestError,
)

# Mock response data
MOCK_RESPONSE = MagicMock(
    choices=[MagicMock(message=MagicMock(content="Test response"))],
)

MOCK_STREAM_RESPONSE = [
    MagicMock(choices=[MagicMock(delta=MagicMock(content="Test "))]),
    MagicMock(choices=[MagicMock(delta=MagicMock(content="stream "))]),
    MagicMock(choices=[MagicMock(delta=MagicMock(content="response"))]),
]


@pytest.fixture
def mock_config():
    """Basic config fixture"""
    return {
        "chat": {
            "system_prompt": "Test system prompt",
            "models": {"available": ["test-model"], "default": "test-model"},
        }
    }


@pytest.fixture
def llm_config(mock_config):
    """LLMConfig fixture"""
    with patch("scraper_chat.core.llm_config.load_config", return_value=mock_config):
        yield LLMConfig("test-model")


@pytest.fixture
def mock_config_with_retries():
    """Config fixture with retry settings"""
    return {
        "chat": {
            "system_prompt": "Test system prompt",
            "models": {"available": ["test-model"], "default": "test-model"},
            "max_retries": 3,
            "retry_base_delay": 0.1,  # Small delays for faster tests
            "retry_max_delay": 0.3,
        }
    }


@pytest.fixture
def llm_config_with_retries(mock_config_with_retries):
    """LLMConfig fixture with retry settings"""
    with patch(
        "scraper_chat.core.llm_config.load_config",
        return_value=mock_config_with_retries,
    ):
        config = LLMConfig("test-model")
        # Ensure we're using the test config values
        config.base_delay = mock_config_with_retries["chat"]["retry_base_delay"]
        config.max_delay = mock_config_with_retries["chat"]["retry_max_delay"]
        yield config


def create_api_error(error_cls, message: str):
    """Helper to create LiteLLM API errors"""
    return error_cls(
        message=message, model="gemini/gemini-1.5-flash", llm_provider="google"
    )


@pytest.mark.asyncio
async def test_get_response_no_stream(llm_config):
    """Test getting a non-streaming response"""
    with patch("litellm.acompletion", return_value=MOCK_RESPONSE):
        response = await llm_config.get_response("test prompt")
        assert response == "Test response"


@pytest.mark.asyncio
async def test_get_response_stream(llm_config):
    """Test getting a streaming response"""
    mock_stream = AsyncMock()
    with patch("litellm.acompletion", return_value=mock_stream):
        response = await llm_config.get_response("test prompt", stream=True)
        assert response == mock_stream


@pytest.mark.asyncio
async def test_get_response_empty_choices(llm_config):
    """Test handling empty choices in response"""
    empty_response = MagicMock(choices=[])
    with patch("litellm.acompletion", return_value=empty_response):
        response = await llm_config.get_response("test prompt")
        assert response == ""  # Should return empty string for empty choices


@pytest.mark.asyncio
async def test_get_response_empty_content(llm_config):
    """Test handling empty content in response"""
    empty_content_response = MagicMock(
        choices=[MagicMock(message=MagicMock(content=""))]
    )
    with patch("litellm.acompletion", return_value=empty_content_response):
        response = await llm_config.get_response("test prompt")
        assert response == ""  # Should return empty string for empty content


@pytest.mark.asyncio
async def test_system_prompt_included(llm_config):
    """Test that system prompt is included in messages"""
    captured_messages = None

    async def mock_acompletion(*args, **kwargs):
        nonlocal captured_messages
        captured_messages = kwargs.get("messages", [])
        return MOCK_RESPONSE

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        await llm_config.get_response("test prompt")
        assert len(captured_messages) == 2
        assert captured_messages[0]["role"] == "system"
        assert captured_messages[0]["content"] == "Test system prompt"


@pytest.mark.asyncio
async def test_retry_timeout(llm_config_with_retries):
    """Test retry behavior on timeout errors"""
    error = create_api_error(Timeout, "Request timed out")

    responses = [error, error, MOCK_RESPONSE]

    async def mock_acompletion(*args, **kwargs):
        if len(responses) > 1:
            result = responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        return responses[0]

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        response = await llm_config_with_retries.get_response("test prompt")
        assert response == "Test response"


@pytest.mark.asyncio
async def test_no_retry_on_bad_request(llm_config_with_retries):
    """Test that bad request errors are not retried"""
    error = create_api_error(BadRequestError, "Bad request")

    with (
        pytest.raises(BadRequestError),
        patch("litellm.acompletion", side_effect=error),
    ):
        await llm_config_with_retries.get_response("test prompt")


@pytest.mark.asyncio
async def test_max_retries_exceeded(llm_config_with_retries):
    """Test that max retries limit is respected"""
    error = create_api_error(Timeout, "Request timed out")

    with (
        pytest.raises(RetryError) as exc_info,
        patch("litellm.acompletion", side_effect=error),
    ):
        await llm_config_with_retries.get_response("test prompt")

    # Verify that the original error is wrapped
    assert isinstance(exc_info.value.last_attempt.exception(), Timeout)
    assert "Request timed out" in str(exc_info.value.last_attempt.exception())


@pytest.mark.asyncio
async def test_retry_with_streaming(llm_config_with_retries):
    """Test retry mechanism works with streaming responses"""
    error = create_api_error(Timeout, "Request timed out")

    mock_stream = AsyncMock()
    responses = [error, error, mock_stream]

    async def mock_acompletion(*args, **kwargs):
        if len(responses) > 1:
            result = responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        return responses[0]

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        response = await llm_config_with_retries.get_response(
            "test prompt", stream=True
        )
        assert response == mock_stream


@pytest.mark.asyncio
async def test_invalid_prompt_type(llm_config):
    """Test that non-string prompts raise TypeError"""
    with pytest.raises(TypeError, match="prompt must be a string"):
        await llm_config.get_response({"invalid": "prompt"})


@pytest.mark.asyncio
async def test_retry_error_message(llm_config_with_retries, caplog):
    """Test that retry error messages are properly logged"""
    error = create_api_error(Timeout, "Request timed out")

    with (
        pytest.raises(RetryError),
        patch("litellm.acompletion", side_effect=error),
        caplog.at_level(logging.WARNING),
    ):
        await llm_config_with_retries.get_response("test prompt")

    # Should see 2 retry attempts in logs (initial attempt + 2 retries = 3 total attempts)
    retry_logs = [
        record for record in caplog.records if "Retrying in" in record.message
    ]
    assert len(retry_logs) == 2

    # Check progressive backoff
    delays = [
        float(log.message.split("Retrying in")[1].split()[0]) for log in retry_logs
    ]
    assert delays[1] > delays[0]  # Second delay should be longer than first

    # Verify attempt numbers
    assert "Attempt 1" in retry_logs[0].message  # First retry after initial attempt
    assert "Attempt 2" in retry_logs[1].message  # Second retry
