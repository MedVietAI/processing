"""
Processing utilities for Vietnamese translation integration
"""

import logging
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger(__name__)

def translate_sft_row(row: Dict[str, Any], translator, text_fields: List[str] = None) -> Dict[str, Any]:
    """
    Translate specific text fields in an SFT row from English to Vietnamese.
    
    Args:
        row: SFT row dictionary
        translator: VietnameseTranslator instance
        text_fields: List of field names to translate. If None, uses default fields.
        
    Returns:
        Translated SFT row dictionary
    """
    if not translator or not translator.is_loaded():
        logger.warning("Translator not available, skipping translation")
        return row
    
    if text_fields is None:
        # Default fields to translate in SFT format
        text_fields = ["instruction", "input", "output"]
    
    try:
        translated_row = translator.translate_dict(row, text_fields)
        logger.debug(f"Translated SFT row with fields: {text_fields}")
        return translated_row
    except Exception as e:
        logger.error(f"Failed to translate SFT row: {e}")
        return row

def translate_rag_row(row: Dict[str, Any], translator, text_fields: List[str] = None) -> Dict[str, Any]:
    """
    Translate specific text fields in a RAG row from English to Vietnamese.
    
    Args:
        row: RAG row dictionary
        translator: VietnameseTranslator instance
        text_fields: List of field names to translate. If None, uses default fields.
        
    Returns:
        Translated RAG row dictionary
    """
    if not translator or not translator.is_loaded():
        logger.warning("Translator not available, skipping translation")
        return row
    
    if text_fields is None:
        # Default fields to translate in RAG format
        text_fields = ["instruction", "input", "output"]
    
    try:
        translated_row = translator.translate_dict(row, text_fields)
        logger.debug(f"Translated RAG row with fields: {text_fields}")
        return translated_row
    except Exception as e:
        logger.error(f"Failed to translate RAG row: {e}")
        return row

def should_translate(vietnamese_translation: bool, translator) -> bool:
    """
    Check if translation should be performed.
    
    Args:
        vietnamese_translation: Flag from user input
        translator: VietnameseTranslator instance
        
    Returns:
        True if translation should be performed
    """
    if not vietnamese_translation:
        return False
    
    if not translator or not translator.is_loaded():
        logger.warning("Vietnamese translation requested but translator not available")
        return False
    
    return True

def log_translation_stats(stats: Dict[str, Any], translated_count: int) -> None:
    """
    Log translation statistics.
    
    Args:
        stats: Statistics dictionary to update
        translated_count: Number of items translated
    """
    stats["vietnamese_translated"] = translated_count
    logger.info(f"Vietnamese translation completed: {translated_count} items translated")