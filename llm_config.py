from typing import Optional, Dict, Any, List
import litellm
import json

# Load config
with open("scraper_config.json", "r") as f:
    config = json.load(f)

class LLMConfig:
    """Simple wrapper for LiteLLM completions"""
    
    def __init__(self, model: str):
        """Initialize with model name"""
        self.model = model
        self.system_prompt = config["chat"]["system_prompt"]
    
    async def get_response(self, messages: List[Dict[str, Any]], stream: bool = False) -> str:
        """Get async response from LLM using message history"""
        # Add system message at the start
        full_messages = [
            {
                "role": "system",
                "content": self.system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ]
        
        # Add message history
        full_messages.extend(messages)
        
        response = litellm.completion(
            model=self.model,
            messages=full_messages,
            stream=stream,
            caching=True
        )
        return response.choices[0].message.content if not stream else response
