import pytest
from unittest.mock import patch
import sys
from pathlib import Path
import gradio as gr

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from scraper_chat.ui.gradio_app import GradioChat, colorize_log


class TestGradioChat:
    @pytest.fixture
    def gradio_chat(self):
        return GradioChat()

    def test_format_plan_mode_references(self, gradio_chat):
        """Test the format_plan_mode_references method"""
        result = gradio_chat.format_plan_mode_references()
        assert result == "📝 **Note:** References are disabled in Plan Mode."

    def test_get_formatted_collections_empty(self, gradio_chat):
        """Test get_formatted_collections with no collections"""
        with patch(
            "scraper_chat.database.chroma_handler.ChromaHandler.get_available_collections",
            return_value=[],
        ):
            result = gradio_chat.get_formatted_collections()
            assert result == []

    def test_get_formatted_collections_with_summary(self, gradio_chat):
        """Test get_formatted_collections with collections having summaries"""
        with (
            patch(
                "scraper_chat.database.chroma_handler.ChromaHandler.get_available_collections",
                return_value=["test_collection"],
            ),
            patch(
                "scraper_chat.database.chroma_handler.ChromaHandler.get_all_documents",
                return_value={"metadatas": [{"summary": "Test summary"}]},
            ),
        ):
            result = gradio_chat.get_formatted_collections()
            assert result == [("📝 test.collection", "test_collection")]

    def test_delete_collection_no_selection(self, gradio_chat):
        """Test delete_collection with no collections selected"""
        result, dropdown = gradio_chat.delete_collection([])
        assert result == "Please select collections to delete"
        assert isinstance(dropdown, gr.Dropdown)

    def test_delete_collection_with_selection(self, gradio_chat):
        """Test delete_collection with collections selected"""
        with (
            patch(
                "scraper_chat.database.chroma_handler.ChromaHandler.delete_collection",
                return_value=True,
            ),
            patch(
                "scraper_chat.database.chroma_handler.ChromaHandler.get_available_collections",
                return_value=[],
            ),
        ):
            result, dropdown = gradio_chat.delete_collection(["collection1"])
            assert "Successfully deleted: collection1" in result
            assert isinstance(dropdown, gr.Dropdown)

    def test_refresh_databases(self, gradio_chat):
        """Test refresh_databases method"""
        with (
            patch(
                "scraper_chat.database.chroma_handler.ChromaHandler.get_available_collections",
                return_value=["test_collection"],
            ),
            patch(
                "scraper_chat.database.chroma_handler.ChromaHandler.get_all_documents",
                return_value={"metadatas": [{"summary": "Test summary"}]},
            ),
        ):
            dropdown, message = gradio_chat.refresh_databases(["test_collection"])
            assert isinstance(dropdown, gr.Dropdown)
            assert message == "Collections refreshed"


def test_colorize_log():
    """Test log colorization for different log levels"""
    # Test error log
    error_log = colorize_log("ERROR: Test error message")
    assert '<span style="color: #ff4444">' in error_log

    # Test warning log
    warning_log = colorize_log("WARNING: Test warning message")
    assert '<span style="color: #ffaa00">' in warning_log

    # Test info log
    info_log = colorize_log("INFO: Test info message")
    assert '<span style="color: #00cc00">' in info_log

    # Test debug log
    debug_log = colorize_log("DEBUG: Test debug message")
    assert '<span style="color: #888888">' in debug_log

    # Test unrecognized log
    plain_log = colorize_log("Test plain message")
    assert plain_log == "Test plain message"
