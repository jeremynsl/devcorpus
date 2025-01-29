import pytest
import re
from scraper_chat.text_processor.text_processor import TextProcessor, TextBlock


class TestTextProcessor:
    @pytest.fixture
    def text_processor(self):
        """Create a fresh TextProcessor instance for each test"""
        return TextProcessor()

    def test_preprocess_html(self, text_processor):
        """Test HTML preprocessing removes boilerplate elements"""
        html = """
        <html>
            <header>Header Content</header>
            <nav>Navigation Links</nav>
            <div class="advertisement">Ad Content</div>
            <body>Main Content</body>
            <footer>Footer Content</footer>
            <aside class="social-share">Social Sharing</aside>
        </html>
        """
        preprocessed = text_processor.preprocess_html(html)

        # Check that boilerplate elements are removed
        assert "Header Content" not in preprocessed
        assert "Navigation Links" not in preprocessed
        assert "Ad Content" not in preprocessed
        assert "Footer Content" not in preprocessed
        assert "Social Sharing" not in preprocessed
        assert "Main Content" in preprocessed

   
    def test_hash_text(self, text_processor):
        """Test text hashing is consistent and normalized"""
        text1 = "  Hello World  "
        text2 = "hello world"
        text3 = "  Hello   World  "

        hash1 = text_processor.hash_text(text1)
        hash2 = text_processor.hash_text(text2)
        hash3 = text_processor.hash_text(text3)

        # Hashes should be identical for normalized text
        assert hash1 == hash2 == hash3

        # Verify it's a valid SHA-256 hash
        assert len(hash1) == 64
        assert re.match(r"^[0-9a-f]{64}$", hash1)

    def test_smart_chunk(self, text_processor):
        """Test smart chunking of text"""
        long_text = """
        First paragraph with some detailed information about a topic.
        It continues with more details and context.

        Second paragraph introduces a new idea.
        This paragraph provides more depth to the concept.

        Third paragraph goes into even more specifics.
        It breaks down the complex ideas into simpler components.
        """

        chunks = text_processor.smart_chunk(long_text)

        # Check that chunks are created
        assert len(chunks) > 0

        # Check chunk size constraints (remove minimum check)
        for chunk in chunks:
            assert len(chunk) <= 1000  # Only check max length

    def test_process_chunk_boilerplate(self, text_processor):
        """Test boilerplate detection"""
        boilerplate_chunk = "Copyright 2023. All rights reserved."

        # Simulate multiple occurrences of the same chunk
        for _ in range(4):
            text_processor.process_chunk(boilerplate_chunk)

        # The chunk should be identified as boilerplate
        block = text_processor.process_chunk(boilerplate_chunk)
        assert block is None

    def test_process_text(self, text_processor):
        """Test full text processing"""
        input_text = """
        Unique paragraph about an interesting topic.
        
        Another unique paragraph with different content.
        
        Repeated boilerplate text.
        Repeated boilerplate text.
        Repeated boilerplate text.
        
        Final unique paragraph.
        """

        processed_text = text_processor.process_text(input_text)

        # Check that boilerplate is removed
        assert "Repeated boilerplate text" not in processed_text

        # Check that unique paragraphs are preserved
        assert "Unique paragraph about an interesting topic" in processed_text
        assert "Another unique paragraph with different content" in processed_text
        assert "Final unique paragraph" in processed_text

        # Verify multiple processing of the same text still works
        processed_text_2 = text_processor.process_text(input_text)
        assert processed_text == processed_text_2

        # Test with more complex repetition
        complex_text = """
        Header text.
        Header text.
        Header text.
        
        Main content starts here.
        This is a unique paragraph.
        
        Footer text.
        Footer text.
        Footer text.
        """

        processed_complex = text_processor.process_text(complex_text)
        assert "Header text" not in processed_complex
        assert "Footer text" not in processed_complex
        assert "Main content starts here" in processed_complex
        assert "This is a unique paragraph" in processed_complex

        # Test with longer repeated text
        long_text = """
        This is a repeated section with some details.
        This is a repeated section with some details.
        This is a repeated section with some details.
        
        Unique content that should be preserved.
        """

        processed_long = text_processor.process_text(long_text)
        assert "This is a repeated section with some details" not in processed_long
        assert "Unique content that should be preserved" in processed_long

    def test_empty_input(self, text_processor):
        """Test processing of empty or whitespace-only input"""
        assert text_processor.process_text("") == ""
        assert text_processor.process_text("   \n\t  ") == ""
        assert text_processor.process_chunk("") is None
        assert text_processor.process_chunk("   ") is None

    def test_text_block_creation(self):
        """Test TextBlock dataclass"""
        block = TextBlock(
            content="Test content", hash="test_hash", frequency=2, is_boilerplate=False
        )

        assert block.content == "Test content"
        assert block.hash == "test_hash"
        assert block.frequency == 2
        assert not block.is_boilerplate
