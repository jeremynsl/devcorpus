"""Advanced text processing module for web scraping."""

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
    """Represents a block of text with metadata."""

    content: str
    hash: str
    frequency: int = 1
    is_boilerplate: bool = False


class TextProcessor:
    """Advanced text processing with boilerplate detection and smart chunking."""

    def __init__(self):
        """Initialize TextProcessor with boilerplate detection settings."""
        self.seen_hashes: Set[str] = set()
        self.block_frequencies: Counter = Counter()
        self.boilerplate_threshold: int = (
            3  # Number of times a block must appear to be considered boilerplate
        )
        self.boilerplate_max_length: int = (
            200  # Maximum length for considering a block as boilerplate
        )

    def preprocess_html(self, html: str) -> str:
        """Remove common boilerplate elements before extraction."""
        soup = BeautifulSoup(html, "html.parser")

        # Remove common boilerplate elements
        for element in soup.find_all(["nav", "footer", "header", "aside"]):
            element.decompose()

        # Remove common advertisement and social media elements
        for element in soup.find_all(
            class_=re.compile(
                r"(ad|advertisement|social|share|cookie|popup|banner|menu)"
            )
        ):
            element.decompose()

        return str(soup)

    def smart_chunk(
        self, text: str, min_chunk_size: int = 100, max_chunk_size: int = 1000
    ) -> List[str]:
        """
        Split text into semantic chunks based on content, treating each line as a separate chunk.
        """
        # Split into sections by paragraph breaks
        sections = re.split(r"\n\s*\n", text)

        chunks = []
        for section in sections:
            # Split section into individual lines
            lines = [line.strip() for line in section.split("\n") if line.strip()]

            for line in lines:
                line_length = len(line)

                # Always treat each line as a separate chunk
                chunks.append(line)

        return chunks

    def hash_text(self, text: str) -> str:
        """Create a normalized hash of text content."""
        # Normalize whitespace and case
        normalized = " ".join(text.split()).lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def process_chunk(self, chunk: str) -> Optional[TextBlock]:
        """Process a single chunk of text."""
        if not chunk.strip():
            return None

        chunk_hash = self.hash_text(chunk)

        # Update frequency counter
        self.block_frequencies[chunk_hash] += 1

        # Check if this is boilerplate
        is_boilerplate = (
            self.block_frequencies[chunk_hash] >= self.boilerplate_threshold
        )

        if is_boilerplate:
            logger.debug(
                f"Detected boilerplate text (frequency={self.block_frequencies[chunk_hash]})"
            )
            return None

        if chunk_hash in self.seen_hashes:
            return None

        self.seen_hashes.add(chunk_hash)
        return TextBlock(chunk, chunk_hash)

    def process_text(
        self, text: str, min_chunk_size: int = 100, max_chunk_size: int = 1000
    ) -> str:
        """
        Process text with smart chunking and boilerplate removal.
        Returns deduplicated and cleaned text.
        """
        # Split text into chunks
        chunks = self.smart_chunk(text, min_chunk_size, max_chunk_size)

        # First pass: calculate chunk frequencies
        frequency_counter = Counter()
        for chunk in chunks:
            normalized = " ".join(chunk.strip().split())
            chunk_hash = self.hash_text(normalized)
            frequency_counter[chunk_hash] += 1

        # Second pass: filter boilerplate and duplicates
        seen_hashes = set()
        processed_blocks = []

        for chunk in chunks:
            normalized = " ".join(chunk.strip().split())
            chunk_hash = self.hash_text(normalized)
            freq = frequency_counter[chunk_hash]

            # Boilerplate check
            is_boilerplate = freq >= self.boilerplate_threshold and (
                len(normalized) < self.boilerplate_max_length
                or re.search(r"^(.+)\1{2,}$", normalized)
            )

            if is_boilerplate:
                continue  # Skip boilerplate

            if chunk_hash not in seen_hashes:
                seen_hashes.add(chunk_hash)
                processed_blocks.append(normalized)

        return "\n\n".join(processed_blocks)
