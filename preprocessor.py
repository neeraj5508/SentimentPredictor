"""
Text preprocessing utilities for sentiment analysis
"""

import re
import string

def preprocess_text(text):
    """
    Preprocess text for sentiment analysis
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Cleaned and preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation but keep spaces
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers (optional - you might want to keep them)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra spaces again after punctuation removal
    text = ' '.join(text.split())
    
    # Strip leading and trailing whitespace
    text = text.strip()
    
    return text

def clean_dataset(texts):
    """
    Clean a list of texts using the preprocess_text function
    
    Args:
        texts (list): List of text strings to preprocess
        
    Returns:
        list: List of preprocessed text strings
    """
    return [preprocess_text(text) for text in texts]

def validate_text_input(text):
    """
    Validate text input for sentiment analysis
    
    Args:
        text (str): Input text to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not text or not isinstance(text, str):
        return False, "Text must be a non-empty string"
    
    # Check if text is too short
    cleaned_text = preprocess_text(text)
    if len(cleaned_text.strip()) < 3:
        return False, "Text is too short for meaningful analysis"
    
    # Check if text is too long (optional limit)
    if len(text) > 10000:
        return False, "Text is too long. Please limit to 10,000 characters"
    
    # Check if text contains only whitespace after preprocessing
    if not cleaned_text.strip():
        return False, "Text contains no meaningful content after preprocessing"
    
    return True, "Valid input"

def get_text_statistics(text):
    """
    Get basic statistics about a text
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary with text statistics
    """
    if not isinstance(text, str):
        return {}
    
    original_length = len(text)
    processed_text = preprocess_text(text)
    processed_length = len(processed_text)
    word_count = len(processed_text.split()) if processed_text else 0
    
    return {
        'original_length': original_length,
        'processed_length': processed_length,
        'word_count': word_count,
        'reduction_ratio': (original_length - processed_length) / original_length if original_length > 0 else 0
    }
