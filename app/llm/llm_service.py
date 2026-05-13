"""
LLM service using Google Gemini's Generative AI API.
Handles AI response generation with retry logic, token management,
and response validation for production robustness.
"""

from typing import Optional
from google import genai
from google.genai import types
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings
from app.utils.logger import tenacity_before_sleep_log


# Global client instance
client = None


def _get_client():
    """Get or initialize the Gemini API client."""
    global client
    if client is None:
        if not settings.GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY is not set in environment!")
        
        # Explicitly use v1 to avoid legacy v1beta issues
        client = genai.Client(
            api_key=settings.GEMINI_API_KEY,
            http_options={"api_version": "v1"}
        )
        logger.info(f"Gemini LLM client initialized (model: {settings.LLM_MODEL})")
    return client


def _make_retry_decorator():
    """Create a retry decorator using current settings."""
    return retry(
        stop=stop_after_attempt(settings.MAX_RETRIES),
        wait=wait_exponential(
            min=settings.RETRY_MIN_WAIT,
            max=settings.RETRY_MAX_WAIT,
        ),
        before_sleep=tenacity_before_sleep_log,
        reraise=True,
    )


@_make_retry_decorator()
def _call_gemini(model_name: str, system_instruction: str, user_prompt: str, temperature: float, max_tokens: int):
    """Make a Gemini generative call with retry logic and fallback."""
    c = _get_client()
    
    try:
        return c.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
    except Exception as e:
        # Check if it's a 404 error and we have a fallback model
        error_str = str(e).lower()
        if "404" in error_str and "not found" in error_str and model_name != settings.FALLBACK_LLM_MODEL:
            logger.warning(f"Model {model_name} not found. Attempting fallback to {settings.FALLBACK_LLM_MODEL}")
            return c.models.generate_content(
                model=settings.FALLBACK_LLM_MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )
        # Reraise if not a 404 or already tried fallback
        raise


def generate(
    system_prompt: str,
    user_prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Generate an AI response using Google Gemini.
    
    Args:
        system_prompt: System-level instructions for the AI
        user_prompt: User's formatted prompt with context
        temperature: Override default temperature
        max_tokens: Override default max token limit
    
    Returns:
        Generated response text
    """
    # Client is initialized lazily in _call_gemini
    temp = temperature if temperature is not None else settings.TEMPERATURE
    tokens = max_tokens if max_tokens is not None else settings.MAX_TOKENS

    try:
        logger.debug(
            f"Gemini request: model={settings.LLM_MODEL}, "
            f"temp={temp}, max_tokens={tokens}, "
            f"system={len(system_prompt)} chars, user={len(user_prompt)} chars"
        )

        response = _call_gemini(settings.LLM_MODEL, system_prompt, user_prompt, temp, tokens)

        # ── Extraction ──────────────────────────────────────
        try:
            result = response.text
        except ValueError:
            # If the response was blocked
            logger.warning(f"Gemini response blocked or empty: {response.prompt_feedback}")
            result = "I'm sorry, I cannot answer that question based on the content policy."

        # ── Usage extraction ────────────────────────────────
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0)
        completion_tokens = getattr(usage, "candidates_token_count", 0)
        total_tokens = getattr(usage, "total_token_count", 0)

        logger.info(
            f"Gemini response: {len(result)} chars, "
            f"tokens(prompt={prompt_tokens}, "
            f"completion={completion_tokens}, "
            f"total={total_tokens})"
        )

        return result

    except Exception as e:
        raise RuntimeError(f"Failed to generate AI response: {str(e)}") from e

