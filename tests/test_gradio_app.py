import pytest
from unittest.mock import patch,MagicMock
import sys
from pathlib import Path
import gradio as gr
import os
import tempfile
import time
import shutil

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from gradio_app import GradioChat, colorize_log, ChromaHandler

class AsyncLLMResponse:
    """Mock LiteLLM streaming response"""
    def __init__(self, chunks):
        self.chunks = chunks
        self.index = 0
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.index >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self.index]
        self.index += 1
        return MagicMock(
            choices=[
                MagicMock(
                    delta=MagicMock(
                        content=chunk
                    )
                )
            ]
        )

class MockChatInterface:
    """Mock chat interface that properly streams responses"""
    def __init__(self, mock_results):
        self.mock_results = mock_results
    
    async def get_response(self, message, return_excerpts=False):
        """Async generator that yields chunks with excerpts"""
        chunks = ["Test ", "response"]
        for chunk in chunks:
            yield (chunk, self.mock_results)

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
def mock_chat_interface():
    mock_results = [{"text": "doc1", "url": "http://test.com", "distance": 0.5}]
    return MockChatInterface(mock_results)

@pytest.fixture
def mock_chroma():
    """Mock ChromaDB handler"""
    handler = MagicMock()
    # Mock get_available_collections to return test collections
    handler.get_available_collections.return_value = ["collection1", "collection2"]
    # Mock query to return empty results
    handler.query.return_value = []
    # Mock delete_collection to return False
    handler.delete_collection.return_value = False
    return handler

@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment"""
    # Configure ChromaDB to use a temporary test directory
    test_db = os.path.join(tempfile.gettempdir(), f"chroma_test_gradio_{int(time.time())}")
    ChromaHandler.configure(test_db)
    
    yield
    
    # Clean up after test
    ChromaHandler._instance = None
    ChromaHandler._client = None
    ChromaHandler._collections = {}
    ChromaHandler.configure(None)  # Reset to default path
    
    if os.path.exists(test_db):
        try:
            shutil.rmtree(test_db)
        except Exception as e:
            print(f"Could not clean up test DB: {e}")

@pytest.fixture
def gradio_chat(mock_config, mock_chat_interface, mock_chroma):
    """Create GradioChat instance with mocks"""
    with patch("gradio_app.config", mock_config), \
         patch("gradio_app.ChromaHandler") as mock_handler_class:
        # Configure mock handler class
        mock_handler_class.get_available_collections.return_value = ["collection1", "collection2"]
        mock_handler_class.return_value = mock_chroma
        
        chat = GradioChat()
        chat.chat_interface = mock_chat_interface
        return chat

@pytest.mark.asyncio
async def test_chat_streaming(gradio_chat):
    """Test chat with streaming responses"""
    history = []
    responses = []
    
    async for new_history, refs in gradio_chat.chat("test message", history, ["collection1"], "test-model"):
        if len(new_history) > 0:
            msg = new_history[-1]["content"]
            if msg:
                responses.append(msg)
    
    # The final response should be the concatenation of all chunks
    assert responses[-1] == "Test response", f"Expected 'Test response', got '{responses[-1]}'"

def test_refresh_databases(gradio_chat):
    """Test refreshing collection list"""
    with patch("gradio_app.ChromaHandler") as mock_handler:
        # Configure mock to return test collections
        mock_handler.get_available_collections.return_value = ["collection1", "collection2"]
        
        dropdown, status = gradio_chat.refresh_databases(["collection1"])
        assert isinstance(dropdown, gr.Dropdown)
        # Handle both tuple and string choices
        choices = []
        for choice in dropdown.choices:
            if isinstance(choice, tuple):
                choices.append(choice[0])
            else:
                choices.append(choice)
        assert sorted(choices) == sorted(["collection1", "collection2"])
        assert "refreshed" in status.lower()

def test_delete_collection_error(gradio_chat):
    """Test error handling in delete collection"""
    with patch("gradio_app.ChromaHandler") as mock_handler_class:
        mock_handler_class.delete_collection.return_value = False
        mock_handler_class.get_available_collections.return_value = ["collection1", "collection2"]
        
        status, dropdown = gradio_chat.delete_collection(["collection1"])
        mock_handler_class.delete_collection.assert_called_once_with("collection1")
        
        assert isinstance(dropdown, gr.Dropdown)
        assert "failed" in status.lower()

@pytest.mark.asyncio
async def test_chat_error_handling(gradio_chat):
    """Test error handling in chat"""
    class ErrorChatInterface:
        async def get_response(self, *args, **kwargs):
            raise Exception("Test error")
            yield  # Make it an async generator
    
    with patch.object(gradio_chat, "chat_interface", ErrorChatInterface()):
        history = []
        error_found = False
        async for new_history, refs in gradio_chat.chat("test message", history, ["collection1"], "test-model"):
            if new_history and len(new_history) > 0:
                if "error" in new_history[-1]["content"].lower():
                    error_found = True
                    break
        assert error_found, "Error message not found in chat response"

def test_colorize_log():
    """Test log colorization"""
    error_log = colorize_log("ERROR: test error")
    warning_log = colorize_log("WARNING: test warning")
    info_log = colorize_log("INFO: test info")
    
    assert '<span style="color: #ff4444">' in error_log
    assert '<span style="color: #ffaa00">' in warning_log
    assert "INFO: test info" in info_log
