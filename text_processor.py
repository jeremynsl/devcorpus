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
        self.seen_hashes: Set[str] = set()
        self.block_frequencies: Counter = Counter()
        self.boilerplate_threshold: int = 3  # Number of times a block must appear to be considered boilerplate
        
    def preprocess_html(self, html: str) -> str:
        """Remove common boilerplate elements before extraction."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove common boilerplate elements
        for element in soup.find_all(['nav', 'footer', 'header', 'aside']):
            element.decompose()
            
        # Remove common advertisement and social media elements
        for element in soup.find_all(class_=re.compile(r'(ad|advertisement|social|share|cookie|popup|banner|menu)')):
            element.decompose()
            
        return str(soup)
    
    def smart_chunk(self, text: str, min_chunk_size: int = 100, max_chunk_size: int = 1000) -> List[str]:
        """
        Split text into semantic chunks based on content.
        Tries to maintain context by keeping related content together.
        """
        # First split by clear section breaks
        sections = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for section in sections:
            # Skip empty sections
            if not section.strip():
                continue
                
            section_size = len(section)
            
            # If section is too big, split it into sentences
            if section_size > max_chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', section)
                for sentence in sentences:
                    sentence_size = len(sentence)
                    
                    # If adding this sentence would exceed max_chunk_size,
                    # save current chunk and start a new one
                    if current_size + sentence_size > max_chunk_size and current_chunk:
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = []
                        current_size = 0
                    
                    current_chunk.append(sentence)
                    current_size += sentence_size
            else:
                # If adding this section would exceed max_chunk_size,
                # save current chunk and start a new one
                if current_size + section_size > max_chunk_size and current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                current_chunk.append(section)
                current_size += section_size
        
        # Add any remaining content
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def hash_text(self, text: str) -> str:
        """Create a normalized hash of text content."""
        # Normalize whitespace and case
        normalized = ' '.join(text.split()).lower()
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def process_chunk(self, chunk: str) -> Optional[TextBlock]:
        """Process a single chunk of text."""
        if not chunk.strip():
            return None
            
        chunk_hash = self.hash_text(chunk)
        
        # Update frequency counter
        self.block_frequencies[chunk_hash] += 1
        
        # Check if this is boilerplate
        is_boilerplate = self.block_frequencies[chunk_hash] >= self.boilerplate_threshold
        
        if is_boilerplate:
            logger.debug(f"Detected boilerplate text (frequency={self.block_frequencies[chunk_hash]})")
            return None
            
        if chunk_hash in self.seen_hashes:
            return None
            
        self.seen_hashes.add(chunk_hash)
        return TextBlock(chunk, chunk_hash)
    
    def process_text(self, text: str, min_chunk_size: int = 100, max_chunk_size: int = 1000) -> str:
        """
        Process text with smart chunking and boilerplate removal.
        Returns deduplicated and cleaned text.
        """
        chunks = self.smart_chunk(text, min_chunk_size, max_chunk_size)
        processed_blocks = []
        
        for chunk in chunks:
            block = self.process_chunk(chunk)
            if block and not block.is_boilerplate:
                processed_blocks.append(block.content)
        
        return '\n\n'.join(processed_blocks)
