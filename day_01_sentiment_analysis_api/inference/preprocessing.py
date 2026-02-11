"""
Text Preprocessing Utilities
Cleans and normalizes text for inference.
"""

import re
import html


def clean_text(text: str) -> str:
    """
    Clean and normalize text for model input.
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned and normalized text
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    
    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)
    
    # Remove @mentions
    text = re.sub(r"@\w+", "", text)
    
    # Remove hashtag symbols (keep the text)
    text = re.sub(r"#(\w+)", r"\1", text)
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def validate_input(text: str) -> tuple[bool, str]:
    """
    Validate input text for the prediction pipeline.
    
    Args:
        text: Input text to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(text, str):
        return False, "Input must be a string"
    
    cleaned = text.strip()
    
    if not cleaned:
        return False, "Input text cannot be empty"
    
    if len(cleaned) < 3:
        return False, "Input text must be at least 3 characters"
    
    if len(cleaned) > 5000:
        return False, "Input text must be under 5000 characters"
    
    return True, ""
