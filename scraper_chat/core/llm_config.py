"""
LLM configuration and rate limiting module.
Provides a wrapper around LiteLLM for making API calls with:
- Configurable rate limiting using a rolling window
- Automatic retries with exponential backoff
- Streaming support
- Error handling and logging
"""

from typing import Union
import litellm
from litellm import (
    Timeout,
    APIConnectionError,
    ServiceUnavailableError,
    InternalServerError,
    RateLimitError,
)
import asyncio
import time
from typing_extensions import AsyncGenerator
from collections import deque
import logging
from scraper_chat.config.config import load_config, CONFIG_FILE
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryCallState,
    RetryError,
)

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter for API calls using a rolling window.

    Implements a token bucket algorithm with epsilon handling for floating-point precision.
    Tracks request times in a rolling window and enforces RPM (requests per minute) limits.
    """

    def __init__(self, rpm: int):
        """
        Initialize rate limiter.

        Args:
            rpm: Maximum requests per minute. Set to 0 to disable rate limiting.
        """
        self.rpm = rpm
        self.window_size = 60  # Seconds
        self.request_times = deque()
        self.lock = asyncio.Lock()
        self.enabled = rpm > 0
        self.epsilon = 1e-6  # For floating-point precision
        self.start_time = time.time()
        self.empty_since = None

    def get_status(self) -> dict:
        """
        Get current rate limiter status.

        Returns:
            Dict containing:
            - rpm: Current requests per minute limit
            - window_remaining: Seconds until oldest request expires
            - active_requests: Number of requests in current window
        """
        now = time.time()
        return {
            "rpm": self.rpm,
            "window_remaining": self.window_size - (now - self.request_times[0])
            if self.request_times
            else 0,
            "active_requests": len(self.request_times),
        }

    def _clean_old_requests(self, now: float) -> None:
        """
        Remove expired requests from the window and update empty state tracking.

        Args:
            now: Current timestamp for comparison
        """
        initial_count = len(self.request_times)

        while self.request_times:
            oldest = self.request_times[0]
            if now - oldest > self.window_size - self.epsilon:
                self.request_times.popleft()
            else:
                break

        if initial_count > 0 and len(self.request_times) == 0:
            self.empty_since = now
            logger.debug("Rate limit buffer emptied - all requests expired")

    async def wait(self) -> None:
        """
        Wait if necessary to comply with rate limits.
        Thread-safe implementation using asyncio lock.
        """
        if not self.enabled:
            return

        async with self.lock:
            now = time.time()
            self._clean_old_requests(now)
            # Calculate oldest request age
            # Build status message
            if len(self.request_times) == 0:
                status_msg = "EMPTY"
                oldest_msg = "N/A"
                if self.empty_since:
                    empty_duration = now - self.empty_since
                    status_msg += f" (for {empty_duration:.1f}s)"
            else:
                status_msg = f"{len(self.request_times)}/{self.rpm}"
                oldest_age = now - self.request_times[0]
                oldest_msg = f"{oldest_age:.1f}s ago"

            print(
                f"[+{now - self.start_time:.1f}s] "
                f"Requests: {status_msg.ljust(15)} | "
                f"Oldest: {oldest_msg}"
            )

            if len(self.request_times) >= self.rpm:
                oldest_age = now - self.request_times[0]
                wait_time = self.window_size - oldest_age + self.epsilon
                logger.debug(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                now = time.time()
                self._clean_old_requests(now)

            self.request_times.append(now)


class LLMConfig:
    """
    Configuration wrapper for LiteLLM API calls.
    Provides retry logic, rate limiting, and response streaming.
    """

    _rate_limiter = None
    _config = None

    def __init__(self, model: str):
        """
        Initialize LLM configuration.

        Args:
            model: Name of the LLM model to use
        """
        self.model = model
        self.config = load_config(CONFIG_FILE)
        self.system_prompt = self.config["chat"]["system_prompt"]
        self.max_retries = self.config.get("chat", {}).get("max_retries", 3)
        self.base_delay = self.config.get("chat", {}).get("retry_base_delay", 1)
        self.max_delay = self.config.get("chat", {}).get("retry_max_delay", 30)

    @classmethod
    def _load_config(cls) -> dict:
        """
        Load configuration file if not already loaded.

        Returns:
            Dictionary containing configuration settings
        """
        if cls._config is None:
            cls._config = load_config(CONFIG_FILE)
        return cls._config

    @classmethod
    def get_rate_limiter(cls) -> RateLimiter:
        """
        Get or create rate limiter instance.

        Returns:
            RateLimiter configured with RPM from config file
        """
        if cls._rate_limiter is None:
            config = cls._load_config()
            rpm = config.get("rate_limit", 0)
            cls._rate_limiter = RateLimiter(rpm)
        return cls._rate_limiter

    @classmethod
    def configure_rate_limit(cls, rpm: int) -> None:
        """
        Configure rate limiting with specified RPM.
        Primarily used for testing.

        Args:
            rpm: Requests per minute limit
        """
        cls._rate_limiter = RateLimiter(rpm)

    def before_sleep_log_message(self, retry_state: RetryCallState) -> None:
        """
        Log message before retry attempt.

        Args:
            retry_state: Current retry state from tenacity
        """
        if retry_state.outcome.failed:
            exception = retry_state.outcome.exception()
            logger.warning(
                f"Attempt {retry_state.attempt_number} failed with {exception.__class__.__name__}. "
                f"Retrying in {retry_state.next_action.sleep} seconds..."
            )

    async def _make_request(
        self, messages: list, stream: bool
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Make LLM API request with retries.

        Args:
            messages: List of message dictionaries
            stream: Whether to stream the response

        Returns:
            Either a complete response string or an async generator for streaming

        Raises:
            RetryError: If all retry attempts fail
        """

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(
                multiplier=self.base_delay, min=self.base_delay, max=self.max_delay
            ),
            retry=retry_if_exception_type(
                (
                    Timeout,
                    APIConnectionError,
                    ServiceUnavailableError,
                    InternalServerError,
                    RateLimitError,
                )
            ),
            before_sleep=lambda retry_state: self.before_sleep_log_message(retry_state),
            reraise=False,
            retry_error_cls=RetryError,
        )
        async def _make_request_with_retry():
            await self.get_rate_limiter().wait()
            return await litellm.acompletion(
                model=self.model, messages=messages, stream=stream
            )

        return await _make_request_with_retry()

    async def get_response(
        self, prompt: str, stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Get response from LLM with retries and rate limiting.

        Args:
            prompt: User's input prompt
            stream: Whether to stream the response

        Returns:
            Either a complete response string or an async generator for streaming

        Raises:
            TypeError: If prompt is not a string
            RetryError: If all retry attempts fail
        """
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")

        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
                "cache_control": {"type": "ephemeral"},
            },
            {"role": "user", "content": prompt, "cache_control": {"type": "ephemeral"}},
        ]

        response = await self._make_request(messages, stream)

        if stream:
            return response

        if not response or not response.choices:
            return ""

        return response.choices[0].message.content

    def get_model(self) -> str:
        """
        Get current model name.

        Returns:
            Name of the LLM model in use
        """
        return self.model

    def set_model(self, model: str):
        """
        Set new model.

        Args:
            model: Name of the LLM model to use
        """
        self.model = model
