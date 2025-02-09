"""Logging configuration module."""

import logging
import sys

LOG_FILE_NAME = "devcorpus.log"
###############################################################################
# Logging Setup
###############################################################################
# One-time configuration at startup

logger = logging.getLogger(__name__)


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_FILE_NAME, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
