"""
Processing utilities for Vietnamese translation integration
"""

import logging
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger(__name__)

def _vi_sanitize_text(s: str) -> str:
    """Light Vietnamese sanitization for finetuning and RAG: strip extra spaces, limit repetition, preserve numbers/units."""
    if not isinstance(s, str):
        return s
    t = s.strip()
    # Collapse repeated punctuation and whitespace
    import re
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"([.?!]){3,}", r"..", t)
    # Remove obvious repetition chunks (very heuristic)
    parts = t.split()
    if len(parts) > 20:
        window = 6
        seen = set()
        filtered = []
        for i in range(len(parts)):
            ngram = " ".join(parts[max(0, i-window):i+1])
            if ngram in seen:
                continue
            seen.add(ngram)
            filtered.append(parts[i])
        t = " ".join(filtered)
    return t

def _validate_vi_translation(original: str, translated: str) -> bool:
    """Validate Vietnamese translation quality"""
    if not translated or not isinstance(translated, str):
        return False
    
    # Check if translation is too short or too different in length
    if len(translated.strip()) < 3:
        return False
    
    # Check if translation contains too much English (should be mostly Vietnamese)
    import re
    english_chars = len(re.findall(r'[a-zA-Z]', translated))
    total_chars = len(re.sub(r'\s', '', translated))
    if total_chars > 0 and english_chars / total_chars > 0.7:
        return False
    
    # Check for common translation failure patterns
    failure_patterns = [
        "translation", "error", "failed", "unable", "cannot",
        "not available", "not found", "invalid", "error"
    ]
    translated_lower = translated.lower()
    for pattern in failure_patterns:
        if pattern in translated_lower:
            return False
    
    return True

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
        # Validate and sanitize translated fields
        for f in text_fields:
            if f in translated_row.get("sft", {}):
                original = row.get("sft", {}).get(f, "")
                translated = translated_row["sft"][f]
                if _validate_vi_translation(original, translated):
                    translated_row["sft"][f] = _vi_sanitize_text(translated)
                else:
                    logger.warning(f"Invalid Vietnamese translation for field {f}, keeping original")
                    translated_row["sft"][f] = original
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
        # Default fields to translate in RAG format (Q, A, C)
        text_fields = ["question", "answer", "context"]
    
    try:
        translated_row = translator.translate_dict(row, text_fields)
        # Validate and sanitize translated fields
        for f in text_fields:
            if f in translated_row:
                original = row.get(f, "")
                translated = translated_row[f]
                if _validate_vi_translation(original, translated):
                    translated_row[f] = _vi_sanitize_text(translated)
                else:
                    logger.warning(f"Invalid Vietnamese translation for field {f}, keeping original")
                    translated_row[f] = original
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