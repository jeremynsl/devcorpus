"""Text processing module for web content with advanced boilerplate detection and deduplication.

This module provides functionality for:
- Smart text chunking based on semantic boundaries
- Boilerplate content detection and removal
- Content deduplication using hash-based comparison
- HTML preprocessing to remove common non-content elements
"""

import re
import hashlib
from typing import List, Set, Optional
from dataclasses import dataclass
from bs4 import BeautifulSoup
import logging
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """A block of text with associated metadata for processing.
    
    Attributes:
        content: The actual text content
        hash: SHA-256 hash of normalized content for comparison
        frequency: Number of times this block appears in the document
        is_boilerplate: Whether this block is identified as boilerplate
    """

    content: str
    hash: str
    frequency: int = 1
    is_boilerplate: bool = False


class TextProcessor:
    """Text processor with boilerplate detection and smart content chunking.
    
    Features:
    - Detects and removes repeated boilerplate content
    - Splits text into semantic chunks while preserving context
    - Deduplicates content using normalized hash comparison
    - Preprocesses HTML to remove non-content elements
    """

    def __init__(self):
        """Initialize processor with default boilerplate detection settings."""
        self.seen_hashes: Set[str] = set()  # For within-document deduplication
        self.global_content_hashes: Set[str] = set()  # For cross-document deduplication
        self.block_frequencies: Counter = Counter()
        self.boilerplate_threshold: int = 5  # Increased from 3 to be less aggressive
        self.boilerplate_max_length: int = 500  # Increased from 200 to allow longer content blocks
        self.current_url: Optional[str] = None  # Track current document URL

    def preprocess_html(self, html: str) -> str:
        """Remove navigation, ads, and other non-content HTML elements.

        Args:
            html: Raw HTML content to process

        Returns:
            Cleaned HTML with common boilerplate elements removed
        """
        soup = BeautifulSoup(html, "html.parser")

        # Only remove obvious non-content elements
        for element in soup.find_all(["nav", "footer", "aside"]):
            element.decompose()

        # Remove ads and popups but be more selective about what we consider boilerplate
        for element in soup.find_all(
            class_=re.compile(
                r"(^ad$|^ads$|advertisement|popup)"
            )
        ):
            element.decompose()

        return str(soup)

    def smart_chunk(
        self, text: str, min_chunk_size: int = 100, max_chunk_size: int = 1000
    ) -> List[str]:
        """Split text into semantic chunks preserving line-based context.

        Args:
            text: Text content to chunk
            min_chunk_size: Minimum size for a chunk in characters
            max_chunk_size: Maximum size for a chunk in characters

        Returns:
            List of text chunks split on semantic boundaries
        """
        # Split on multiple blank lines to preserve paragraphs
        sections = re.split(r"\n\s*\n\s*\n", text)
        chunks = []
        current_chunk = []
        current_size = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue

            section_size = len(section)

            # If section is too big, split it further on single blank lines
            if section_size > max_chunk_size:
                subsections = re.split(r"\n\s*\n", section)
                for subsection in subsections:
                    subsection = subsection.strip()
                    if not subsection:
                        continue
                    chunks.append(subsection)
            else:
                chunks.append(section)

        return chunks

    def hash_text(self, text: str) -> str:
        """Generate normalized SHA-256 hash of text content.

        Args:
            text: Text to hash

        Returns:
            Hex digest of normalized text hash
        """
        # Normalize whitespace but preserve some structure
        normalized = re.sub(r'\s+', ' ', text).strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def process_chunk(self, chunk: str) -> Optional[TextBlock]:
        """Process a single chunk of text, detecting boilerplate and duplicates.

        Args:
            chunk: Text chunk to process

        Returns:
            TextBlock if chunk is unique and not boilerplate, None otherwise
        """
        if not chunk.strip():
            return None

        chunk_hash = self.hash_text(chunk)
        
        # Only count frequency for shorter chunks that might be boilerplate
        if len(chunk) <= self.boilerplate_max_length:
            self.block_frequencies[chunk_hash] += 1
            
            # Mark as boilerplate if it appears too frequently and is relatively short
            is_boilerplate = self.block_frequencies[chunk_hash] >= self.boilerplate_threshold
            if is_boilerplate:
                logger.debug(
                    f"Detected boilerplate text (frequency={self.block_frequencies[chunk_hash]}, length={len(chunk)})"
                )
                return None

        if chunk_hash in self.seen_hashes:
            return None

        self.seen_hashes.add(chunk_hash)
        return TextBlock(chunk, chunk_hash)

    def process_text(
        self, text: str, min_chunk_size: int = 100, max_chunk_size: int = 1000, url: Optional[str] = None
    ) -> str:
        """Process text with smart chunking, boilerplate removal and deduplication.

        Args:
            text: Text content to process
            min_chunk_size: Minimum size for text chunks
            max_chunk_size: Maximum size for text chunks
            url: Optional URL of the document being processed, used for cross-document deduplication

        Returns:
            Processed text with boilerplate and duplicates removed
        """
        # Only do global deduplication if we have a URL
        if url is not None:
            # Instead of hashing the entire document, hash significant chunks
            chunks = self.smart_chunk(text)
            significant_chunks = [c for c in chunks if len(c) > self.boilerplate_max_length]
            
            # If we find that most significant chunks are duplicates, consider the document a duplicate
            if significant_chunks:
                duplicate_count = 0
                for chunk in significant_chunks:
                    chunk_hash = self.hash_text(chunk)
                    if chunk_hash in self.global_content_hashes:
                        duplicate_count += 1
                
                # If more than 70% of significant chunks are duplicates, skip this document
                if duplicate_count / len(significant_chunks) > 0.7 and url != self.current_url:
                    logger.info(f"Skipping document with {duplicate_count}/{len(significant_chunks)} duplicate chunks from {url}")
                    return ""
            
            # Add chunk hashes to global set
            for chunk in significant_chunks:
                self.global_content_hashes.add(self.hash_text(chunk))
            
            self.current_url = url
        
        # Reset seen_hashes for this document
        self.seen_hashes.clear()
        self.block_frequencies.clear()
        
        chunks = self.smart_chunk(text, min_chunk_size, max_chunk_size)
        processed_blocks = []

        for chunk in chunks:
            block = self.process_chunk(chunk)
            if block:
                processed_blocks.append(block.content)

        return "\n\n".join(processed_blocks)
