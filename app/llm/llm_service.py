"""
LLM service using Google Gemini's Generative AI API.
Handles AI response generation with retry logic, token management,
and response validation for production robustness.
"""

from typing import Optional
from google import genai
from google.genai import types
from loguru import logger
from functools import lru_cache
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

from app.config import settings
from app.utils.logger import tenacity_before_sleep_log
from app.utils.gemini_key_manager import get_key_manager


def _get_client():
    """Get the Gemini API client via key manager."""
    return get_key_manager().get_client(api_version="v1")


def _is_retryable_exception(e):
    """Check if the exception should trigger a retry."""
    err_str = str(e).lower()
    # Retry on 429 (rate limit) and 5xx (server errors)
    return "429" in err_str or "quota" in err_str or "exhausted" in err_str or "500" in err_str or "503" in err_str

def _make_retry_decorator():
    """Create a retry decorator using current settings."""
    return retry(
        stop=stop_after_attempt(settings.MAX_RETRIES),
        wait=wait_exponential(
            min=settings.RETRY_MIN_WAIT,
            max=settings.RETRY_MAX_WAIT,
        ),
        retry=retry_if_exception(_is_retryable_exception),
        before_sleep=tenacity_before_sleep_log,
        reraise=True,
    )


@_make_retry_decorator()
def _call_gemini(model_name: str, system_instruction: str, user_prompt: str, temperature: float, max_tokens: int):
    """Make a Gemini generative call with retry logic, key rotation, and fallback."""
    km = get_key_manager()
    c = km.get_client(api_version="v1")
    
    # Prepare system instruction as a Content object for better SDK compatibility
    # and to avoid 'systemInstruction' name mismatch issues in some API versions.
    sys_inst = types.Content(
        parts=[types.Part(text=system_instruction)],
        role="system"
    )

    try:
        return c.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_inst,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
    except Exception as e:
        error_str = str(e).lower()
        
        # ── Key rotation on 429 / quota errors ──────────────
        if km.is_quota_error(e):
            if km.rotate_key():
                logger.warning("Quota/rate-limit hit — rotated to next API key, retrying...")
                new_client = km.get_client(api_version="v1")
                return new_client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=sys_inst,
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    )
                )
            # All keys exhausted — let tenacity / existing error handling deal with it
            raise

        # Handle 404 Model Not Found
        if "404" in error_str and "not found" in error_str and model_name != settings.FALLBACK_LLM_MODEL:
            logger.warning(f"Model {model_name} not found. Attempting fallback to {settings.FALLBACK_LLM_MODEL}")
            return c.models.generate_content(
                model=settings.FALLBACK_LLM_MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=sys_inst,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )
            
        # Handle 400 Invalid Argument (like the systemInstruction issue)
        if "400" in error_str and "invalid" in error_str:
            logger.warning("Invalid argument in request. Retrying without explicit system_instruction field...")
            # Fallback: Merge system prompt into user prompt if the API rejected the field
            merged_prompt = f"{system_instruction}\n\nUSER REQUEST:\n{user_prompt}"
            return c.models.generate_content(
                model=model_name,
                contents=merged_prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )
            
        # Reraise if not handled
        raise


@lru_cache(maxsize=128)
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
        error_str = str(e).lower()
        if "429" in error_str or "quota" in error_str:
            logger.warning("Gemini quota exhausted. Returning rate-limit message to user.")
            return "I'm sorry, but the AI service is currently at its daily limit. Please try again later or ask a question that doesn't require complex generation."
        
        raise RuntimeError(f"Failed to generate AI response: {str(e)}") from e

