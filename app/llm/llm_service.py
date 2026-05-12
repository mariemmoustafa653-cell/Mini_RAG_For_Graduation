"""
LLM service using OpenAI's chat completion API.
Handles AI response generation with retry logic, token management,
and response validation for production robustness.
"""

from typing import Optional

from openai import OpenAI
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from app.config import settings


# Module-level client (initialized lazily)
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Get or create the OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info(f"OpenAI LLM client initialized (model: {settings.LLM_MODEL})")
    return _client


def _make_retry_decorator():
    """Create a retry decorator using current settings."""
    return retry(
        stop=stop_after_attempt(settings.MAX_RETRIES),
        wait=wait_exponential(
            min=settings.RETRY_MIN_WAIT,
            max=settings.RETRY_MAX_WAIT,
        ),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )


@_make_retry_decorator()
def _call_openai(client: OpenAI, messages: list[dict], temperature: float, max_tokens: int):
    """Make an OpenAI chat completion call with retry logic."""
    return client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.9,
    )


def generate(
    system_prompt: str,
    user_prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Generate an AI response using OpenAI chat completion.
    
    Includes retry logic for transient API failures,
    configurable generation parameters, and response validation.
    
    Args:
        system_prompt: System-level instructions for the AI
        user_prompt: User's formatted prompt with context
        temperature: Override default temperature (lower = more factual)
        max_tokens: Override default max token limit
    
    Returns:
        Generated response text
    """
    client = _get_client()
    temp = temperature if temperature is not None else settings.TEMPERATURE
    tokens = max_tokens if max_tokens is not None else settings.MAX_TOKENS

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        logger.debug(
            f"LLM request: model={settings.LLM_MODEL}, "
            f"temp={temp}, max_tokens={tokens}, "
            f"system={len(system_prompt)} chars, user={len(user_prompt)} chars"
        )

        response = _call_openai(client, messages, temp, tokens)

        result = response.choices[0].message.content
        usage = response.usage

        # ── Response validation ──────────────────────────────
        if not result or not result.strip():
            logger.warning("LLM returned empty response — using fallback")
            result = "I was unable to generate a response. Please try rephrasing your question."

        # ── Check for refusal ────────────────────────────────
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "content_filter":
            logger.warning("LLM response was filtered by content policy")
            result = "The response was filtered by content policy. Please rephrase your question."

        logger.info(
            f"LLM response: {len(result)} chars, "
            f"tokens(prompt={usage.prompt_tokens}, "
            f"completion={usage.completion_tokens}, "
            f"total={usage.total_tokens}), "
            f"finish_reason={finish_reason}"
        )

        return result

    except Exception as e:
        logger.error(f"LLM generation error after retries: {e}")
        raise RuntimeError(f"Failed to generate AI response: {e}") from e
