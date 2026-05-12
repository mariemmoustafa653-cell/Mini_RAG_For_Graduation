"""
Arabic text processing utilities.
Handles normalization, diacritics removal, and language detection.
"""

import re
import unicodedata


# Arabic Unicode ranges
_ARABIC_RANGE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")

# Arabic diacritics (tashkeel)
_ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]"
)

# Alef variations → normalized Alef
_ALEF_VARIATIONS = re.compile(r"[\u0622\u0623\u0625]")  # آ أ إ → ا

# Taa Marbuta → Haa
_TAA_MARBUTA = "\u0629"  # ة
_HAA = "\u0647"  # ه


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for consistent embedding and retrieval.
    
    Steps:
    1. Remove diacritics (tashkeel)
    2. Normalize Alef variations (آ أ إ → ا)
    3. Normalize Taa Marbuta (ة → ه)
    4. Normalize Unicode (NFC form)
    """
    if not text:
        return text

    # Remove diacritics
    text = _ARABIC_DIACRITICS.sub("", text)

    # Normalize Alef
    text = _ALEF_VARIATIONS.sub("\u0627", text)

    # Normalize Taa Marbuta
    text = text.replace(_TAA_MARBUTA, _HAA)

    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    return text


def detect_language(text: str) -> str:
    """
    Detect whether text is primarily Arabic or English.
    
    Returns:
        "ar" if Arabic characters dominate, "en" otherwise.
    """
    if not text:
        return "en"

    arabic_chars = len(_ARABIC_RANGE.findall(text))
    total_alpha = sum(1 for c in text if c.isalpha())

    if total_alpha == 0:
        return "en"

    arabic_ratio = arabic_chars / total_alpha
    return "ar" if arabic_ratio > 0.3 else "en"


def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing.
    
    Steps:
    1. Remove control characters (except newlines and tabs)
    2. Normalize whitespace
    3. Strip leading/trailing whitespace
    """
    if not text:
        return text

    # Remove control characters except \n and \t
    text = re.sub(r"[^\S\n\t ]+", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Normalize multiple spaces to single space
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize multiple newlines to double newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()
