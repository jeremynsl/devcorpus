import logging
import sys

LOG_FILE_NAME = "scraper.log"
###############################################################################
# Logging Setup
###############################################################################
logger = logging.getLogger("RecursiveScraper")
logger.setLevel(logging.DEBUG)

log_formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)

# File handler
file_handler = logging.FileHandler(LOG_FILE_NAME, mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(log_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)