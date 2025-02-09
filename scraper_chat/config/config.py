"""Config module for DevCorpus."""

import os
import sys
import json
import logging

logger = logging.getLogger(__name__)

CONFIG_FILE = "scraper_config.json"


def load_config(config_path: str) -> dict:  # Return full config as dictionary
    """Load and return the entire config as a dictionary"""
    if not os.path.exists(config_path):
        logger.error(f"Config file {config_path} not found.")
        sys.exit(1)
    ## validate json config

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except ValueError as e:
            logger.error(f"Invalid JSON config file {config_path}: {e}")
            sys.exit(1)


def save_config(new_config, config_path: str):
    with open(config_path, "w") as f:
        json.dump(new_config, f, indent=4)
