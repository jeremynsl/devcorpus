import os
import sys
import json
from logger import logger

###############################################################################
# Global Config / Constants
###############################################################################
CONFIG_FILE = "scraper_config.json"







###############################################################################
# Load Config
###############################################################################
def load_config(config_path: str):
    """
    Load the JSON config file, which should contain:
        {
          "proxies": ["http://proxy1:8080", "http://proxy2:8080"],
          "rate_limit": 5,
          "user_agent": "MyScraperBot/1.0"
        }
    """
    if not os.path.exists(config_path):
        logger.error(f"Config file {config_path} not found.")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    proxies = data.get("proxies", [])
    rate_limit = data.get("rate_limit", 5)
    user_agent = data.get("user_agent", "MyScraperBot/1.0")

    return proxies, rate_limit, user_agent