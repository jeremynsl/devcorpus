import re
import hashlib
import logging
from typing import List, Set, Optional
from dataclasses import dataclass
from collections import Counter
from bs4 import BeautifulSoup
import difflib
import nltk
from nltk.tokenize import sent_tokenize


nltk.download("punkt", quiet=True)

logger = logging.getLogger(__name__)


def sentence_split(text: str) -> List[str]:
    return sent_tokenize(text)


@dataclass
class TextBlock:
    """
    A block of text with associated metadata for processing.
    """

    content: str
    hash: str
    frequency: int = 1
    is_boilerplate: bool = False


class TextProcessor:
    """
    Text processor with enhanced boilerplate detection, fuzzy deduplication,
    and semantic chunking.
    """

    def __init__(self):
        # For within-document deduplication.
        self.seen_hashes: Set[str] = set()
        # For cross-document deduplication.
        self.global_content_hashes: Set[str] = set()
        self.block_frequencies: Counter = Counter()

        # Adjustable thresholds and parameters.
        self.boilerplate_threshold: int = (
            5  # Appearances before flagging as boilerplate.
        )
        self.boilerplate_max_length: int = (
            500  # Only short blocks (<= this) are checked for repetition.
        )
        self.fuzzy_similarity_threshold: float = (
            0.95  # Fuzzy matching similarity threshold.
        )
        self.current_url: Optional[str] = None  # Track current document URL.

    def preprocess_html(self, html: str) -> str:
        """
        Remove scripts, styles, comments, and non-content elements.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style tags.
        for tag in soup(["script", "style"]):
            tag.decompose()

        # Remove HTML comments.
        for comment in soup.find_all(
            string=lambda text: isinstance(text, type(soup.Comment))
        ):
            comment.extract()

        # Remove structural elements likely to be boilerplate.
        for element in soup.find_all(["nav", "footer", "aside", "header"]):
            element.decompose()

        # Remove ads/popups with common class patterns.
        for element in soup.find_all(
            class_=re.compile(r"(^ad$|^ads$|advertisement|popup|banner|sponsored)")
        ):
            element.decompose()

        return str(soup)

    def smart_chunk(
        self, text: str, min_chunk_size: int = 100, max_chunk_size: int = 1000
    ) -> List[str]:
        """
        Split text into semantic chunks:
          1. Split by paragraphs (double newlines).
          2. Remove paragraphs that consist entirely of repeated lines.
          3. For remaining paragraphs longer than max_chunk_size, split on sentence boundaries.
          4. Merge chunks that are too short.
        """
        # Split into paragraphs and filter out empty ones
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        # Filter out paragraphs that consist entirely of repeated lines
        filtered_paragraphs = []
        for para in paragraphs:
            lines = [line.strip() for line in para.split("\n") if line.strip()]
            # Only keep paragraphs that either have one line or have different lines
            if len(lines) <= 1 or len(set(lines)) > 1:
                filtered_paragraphs.append(para)

        chunks = []
        # Process each filtered paragraph
        for para in filtered_paragraphs:
            if len(para) <= max_chunk_size:
                chunks.append(para)
            else:
                sentences = sentence_split(para)
                current_chunk = ""
                for sent in sentences:
                    # If adding the sentence would exceed max_chunk_size, start a new chunk
                    if len(current_chunk) + len(sent) + 1 > max_chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sent
                    else:
                        current_chunk = (
                            f"{current_chunk} {sent}" if current_chunk else sent
                        )
                if current_chunk:
                    chunks.append(current_chunk.strip())

        # Merge adjacent chunks if they are below the minimum size
        merged_chunks = []
        buffer = ""
        for chunk in chunks:
            if len(buffer) < min_chunk_size:
                buffer = f"{buffer} {chunk}".strip() if buffer else chunk
            else:
                merged_chunks.append(buffer)
                buffer = chunk
        if buffer:
            merged_chunks.append(buffer)

        return merged_chunks

    def hash_text(self, text: str) -> str:
        """
        Generate a normalized SHA-256 hash of the text.
        """
        normalized = re.sub(r"\s+", " ", text).strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def is_fuzzy_duplicate(self, new_text: str, existing_text: str) -> bool:
        """
        Determine if two text blocks are nearly identical using fuzzy matching.
        Optimizations:
          - Skip if lengths differ by more than 20%.
          - Use SequenceMatcher.quick_ratio() as a fast pre-check.
        """
        len_new = len(new_text)
        len_exist = len(existing_text)
        # Skip fuzzy matching if the lengths differ significantly.
        if abs(len_new - len_exist) > 0.2 * max(len_new, len_exist):
            return False

        sm = difflib.SequenceMatcher(None, new_text, existing_text)
        # Use quick_ratio as a preliminary check.
        if sm.quick_ratio() < self.fuzzy_similarity_threshold:
            return False

        ratio = sm.ratio()
        return ratio >= self.fuzzy_similarity_threshold

    def _is_repeated_lines_chunk(self, chunk: str) -> bool:
        """
        Check if a chunk consists entirely of repeated lines.
        Returns True if all non-empty lines in the chunk are identical and there's more than one line.
        """
        lines = [line.strip() for line in chunk.split("\n") if line.strip()]
        if not lines:
            return False
        return len(lines) > 1 and len(set(lines)) == 1

    def process_chunk(
        self, chunk: str, processed_blocks: List[TextBlock]
    ) -> Optional[TextBlock]:
        """
        Process a single chunk of text:
          - Apply boilerplate filtering.
          - Perform exact and fuzzy deduplication.
        """
        if not chunk.strip():
            return None

        # Normalize the chunk for frequency counting
        normalized_chunk = re.sub(r"\s+", " ", chunk.strip())
        chunk_hash = self.hash_text(chunk)

        # Check for boilerplate based on frequency (only for shorter chunks).
        if len(normalized_chunk) <= self.boilerplate_max_length:
            self.block_frequencies[normalized_chunk] += 1
            if self.block_frequencies[normalized_chunk] >= self.boilerplate_threshold:
                logger.debug(
                    f"Detected boilerplate text (frequency={self.block_frequencies[normalized_chunk]}, length={len(normalized_chunk)})"
                )
                return None

        if chunk_hash in self.seen_hashes:
            return None

        # Fuzzy duplicate check: compare with all previously accepted blocks.
        for block in processed_blocks:
            if self.is_fuzzy_duplicate(chunk, block.content):
                logger.debug("Fuzzy duplicate detected; skipping similar block.")
                return None

        self.seen_hashes.add(chunk_hash)
        return TextBlock(content=chunk, hash=chunk_hash)

    def process_text(
        self,
        text: str,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        url: Optional[str] = None,
    ) -> str:
        """
        Process text by performing:
          1. Cross-document deduplication (if URL is provided).
          2. Smart chunking.
          3. Boilerplate and duplicate removal.
        """
        if not text.strip():
            return ""

        # Reset per-document caches
        self.seen_hashes.clear()
        self.block_frequencies.clear()
        processed_blocks: List[TextBlock] = []

        # Call smart_chunk once and reuse its output
        chunks = self.smart_chunk(text, min_chunk_size, max_chunk_size)
        if not chunks:
            return ""

        # Global deduplication based on content similarity
        if url is not None and url != self.current_url:
            # Calculate hash of entire normalized content for exact matching
            normalized_text = re.sub(r"\s+", " ", text.strip())
            full_content_hash = self.hash_text(normalized_text)

            # First check if we've seen this exact content before
            if full_content_hash in self.global_content_hashes:
                logger.info(f"Skipping duplicate document from {url}")
                return ""

            # For partial matching, check significant chunks
            significant_chunks = [
                c for c in chunks if len(c) > self.boilerplate_max_length
            ]
            if significant_chunks:
                chunk_hashes = {self.hash_text(chunk) for chunk in significant_chunks}
                duplicate_hashes = chunk_hashes.intersection(self.global_content_hashes)

                if (
                    len(duplicate_hashes) / len(chunk_hashes) > 0.75
                ):  # Lower threshold for better detection
                    logger.info(
                        f"Skipping similar document from {url} "
                        f"({len(duplicate_hashes)}/{len(chunk_hashes)} significant chunks duplicated)."
                    )
                    return ""

                # Add both chunk hashes and full content hash to global set
                self.global_content_hashes.update(chunk_hashes)

            # Always add the full content hash if we're keeping the document
            self.global_content_hashes.add(full_content_hash)
            self.current_url = url

        # Pre-process chunks to detect boilerplate patterns
        chunk_frequencies = Counter()
        for chunk in chunks:
            normalized = re.sub(r"\s+", " ", chunk.strip())
            if len(normalized) <= self.boilerplate_max_length:
                chunk_frequencies[normalized] += 1

        # Process each chunk individually, with awareness of global frequencies
        for chunk in chunks:
            normalized = re.sub(r"\s+", " ", chunk.strip())
            if len(normalized) <= self.boilerplate_max_length:
                self.block_frequencies[normalized] = chunk_frequencies[normalized]

            block = self.process_chunk(chunk, processed_blocks)
            if block:
                processed_blocks.append(block)

        logger.info(f"Processed {len(processed_blocks)} blocks from the document.")
        return "\n\n".join(block.content for block in processed_blocks)
