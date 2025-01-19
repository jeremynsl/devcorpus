from typing import Union
import litellm
import json
import asyncio
import time
from typing_extensions import AsyncGenerator
from collections import deque

# Load config
with open("scraper_config.json", "r") as f:
    config = json.load(f)

class RateLimiter:
    """Rate limiter for API calls using a rolling window"""
    def __init__(self, rpm: int):
        self.rpm = rpm
        self.window_size = 60  # Window size in seconds
        self.request_times = deque()  # Track timestamps of requests
        self.lock = asyncio.Lock()
        self.enabled = rpm > 0  # Disable if rpm is 0 or negative
        
    def _clean_old_requests(self, now: float):
        """Remove requests older than the window size"""
        while self.request_times and now - self.request_times[0] >= self.window_size:
            self.request_times.popleft()
            
    async def wait(self):
        """Wait if needed to maintain rate limit"""
        if not self.enabled:
            return
            
        async with self.lock:
            now = time.time()
            
            # Clean old requests
            self._clean_old_requests(now)
            
            # If we're at the limit, wait until oldest request expires
            if len(self.request_times) >= self.rpm:
                wait_time = self.window_size - (now - self.request_times[0])
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    now = time.time()  # Update current time after waiting
                    self._clean_old_requests(now)  # Clean again after waiting
            
            # Add current request
            self.request_times.append(now)

class LLMConfig:
    """Simple wrapper for LiteLLM completions"""
    _rate_limiter = None  # Class-level rate limiter
    
    @classmethod
    def get_rate_limiter(cls):
        """Get or create the rate limiter"""
        if cls._rate_limiter is None:
            # Check if we're in test mode (indicated by rpm <= 0)
            rpm = config["chat"].get("rate_limit_rpm", 9)
            cls._rate_limiter = RateLimiter(rpm)
        return cls._rate_limiter
    
    @classmethod
    def configure_rate_limit(cls, rpm: int):
        """Configure rate limiting, mainly for testing"""
        cls._rate_limiter = RateLimiter(rpm)
    
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
        
        # Wait for rate limit before making request
        await self.get_rate_limiter().wait()
        
        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            stream=stream
        )
        
        if stream:
            return response  # Return AsyncGenerator for streaming
        else:
            return response.choices[0].message.content if response.choices else ""
