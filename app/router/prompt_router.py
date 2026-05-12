"""
Prompt routing system.
Maps AI actions to their corresponding prompt templates
and formats them with context and user messages.
"""

from loguru import logger

from app.prompts.templates import PROMPT_TEMPLATES, SYSTEM_PROMPT, SUPPORTED_ACTIONS


class UnsupportedActionError(Exception):
    """Raised when an unsupported action is requested."""
    pass


def route(action: str, context: str, message: str) -> tuple[str, str]:
    """
    Route an action to the appropriate prompt template.
    
    Args:
        action: The AI action to perform (chat, summarize, quiz, etc.)
        context: Retrieved document context
        message: User's message or question
    
    Returns:
        Tuple of (system_prompt, user_prompt)
    
    Raises:
        UnsupportedActionError: If action is not supported
    """
    action = action.lower().strip()

    if action not in PROMPT_TEMPLATES:
        raise UnsupportedActionError(
            f"Unsupported action: '{action}'. "
            f"Supported actions: {', '.join(SUPPORTED_ACTIONS)}"
        )

    template = PROMPT_TEMPLATES[action]
    user_prompt = template.format(context=context, message=message)

    logger.info(f"Routed action '{action}' → prompt ({len(user_prompt)} chars)")
    return SYSTEM_PROMPT, user_prompt


def get_supported_actions() -> list[str]:
    """Return list of supported AI actions."""
    return SUPPORTED_ACTIONS.copy()
