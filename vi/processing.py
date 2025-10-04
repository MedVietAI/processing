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
    
    # Check if translation is too short
    if len(translated.strip()) < 3:
        return False
    
    # If translation is identical to original, it's not a valid translation
    if translated.strip() == original.strip():
        return False
    
    # Check for common translation failure patterns
    failure_patterns = [
        "translation error", "translation failed", "unable to translate", 
        "cannot translate", "not available", "not found", "invalid translation"
    ]
    translated_lower = translated.lower()
    for pattern in failure_patterns:
        if pattern in translated_lower:
            return False
    
    # Check if translation contains Vietnamese characters (basic check)
    import re
    vietnamese_chars = len(re.findall(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', translated, re.IGNORECASE))
    total_chars = len(re.sub(r'\s', '', translated))
    
    # If there are Vietnamese characters, it's likely a valid translation
    if vietnamese_chars > 0:
        return True
    
    # If no Vietnamese characters but significantly different from original, accept it
    # (some translations might not have Vietnamese diacritics)
    if len(translated) > len(original) * 0.5 and len(translated) < len(original) * 2.0:
        return True
    
    return False

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
        # Create a copy of the row to avoid modifying the original
        translated_row = row.copy()
        
        # Translate the SFT fields directly
        sft_data = row.get("sft", {})
        translated_sft = {}
        
        for field in text_fields:
            if field in sft_data and isinstance(sft_data[field], str) and sft_data[field].strip():
                try:
                    original = sft_data[field]
                    translated = translator.translate_text(original)
                    
                    # Validate and sanitize translated field
                    if _validate_vi_translation(original, translated):
                        translated_sft[field] = _vi_sanitize_text(translated)
                        logger.debug(f"Translated field '{field}': '{original[:50]}...' -> '{translated[:50]}...'")
                    else:
                        logger.warning(f"Invalid Vietnamese translation for field {field}, keeping original")
                        translated_sft[field] = original
                except Exception as e:
                    logger.error(f"Failed to translate field '{field}': {e}")
                    translated_sft[field] = sft_data[field]
            else:
                # Keep original if field doesn't exist or is empty
                translated_sft[field] = sft_data.get(field, "")
        
        # Update the translated row
        translated_row["sft"] = translated_sft
        
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