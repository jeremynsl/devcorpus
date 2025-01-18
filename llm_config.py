from typing import Union
import litellm
import json
from typing_extensions import AsyncGenerator

# Load config
with open("scraper_config.json", "r") as f:
    config = json.load(f)

class LLMConfig:
    """Simple wrapper for LiteLLM completions"""
    
    def __init__(self, model: str):
        """Initialize with model name"""
        self.model = model
        self.system_prompt = config["chat"]["system_prompt"]
    
    async def get_response(self, prompt: str, stream: bool = False) -> Union[str, AsyncGenerator[str, None]]:
        """Get async response from LLM"""
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
                "cache_control": {"type": "ephemeral"}
            },
            {
                "role": "user",
                "content": prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ]
        
        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            stream=stream
        )
        
        if stream:
            return response  # Return AsyncGenerator for streaming
        else:
            return response.choices[0].message.content if response.choices else ""
