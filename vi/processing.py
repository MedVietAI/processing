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

def _is_vietnamese_text(text: str) -> bool:
    """Check if text is already in Vietnamese"""
    if not text or not isinstance(text, str):
        return False
    
    import re
    # Check for Vietnamese characters
    vietnamese_chars = len(re.findall(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', text, re.IGNORECASE))
    total_chars = len(re.sub(r'\s', '', text))
    
    # If more than 20% of characters are Vietnamese, consider it Vietnamese text
    if total_chars > 0 and vietnamese_chars / total_chars > 0.2:
        return True
    
    # Check for common Vietnamese words (including single words)
    vietnamese_words = ['chào', 'xin chào', 'cảm ơn', 'tôi', 'bạn', 'là', 'có', 'không', 'và', 'của', 'trong', 'với', 'để', 'cho', 'về', 'từ', 'đến', 'tại', 'này', 'đó', 'đây', 'kia', 'nào', 'sao', 'thế', 'nào', 'gì', 'ai', 'đâu', 'khi', 'nếu', 'mà', 'để', 'cho', 'về', 'từ', 'đến', 'tại', 'triệu', 'chứng', 'bệnh', 'tiểu', 'đường', 'bác', 'sĩ', 'bệnh', 'nhân']
    text_lower = text.lower()
    vietnamese_word_count = sum(1 for word in vietnamese_words if word in text_lower)
    
    # If text contains any Vietnamese words, consider it Vietnamese
    if vietnamese_word_count >= 1:
        return True
    
    return False

def _validate_vi_translation(original: str, translated: str) -> bool:
    """Validate Vietnamese translation quality"""
    if not translated or not isinstance(translated, str):
        return False
    
    # Check if translation is too short
    if len(translated.strip()) < 3:
        return False
    
    # If translation is identical to original, check if original was already Vietnamese
    if translated.strip() == original.strip():
        # If original was already Vietnamese, this is actually a valid case
        if _is_vietnamese_text(original):
            return True
        # Otherwise, it's not a valid translation
        return False
    
    # Check if original was Vietnamese but translated is English (wrong direction)
    if _is_vietnamese_text(original) and not _is_vietnamese_text(translated):
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
                    
                    # Check if text is already in Vietnamese - skip translation if so
                    if _is_vietnamese_text(original):
                        logger.debug(f"Field '{field}' is already in Vietnamese, skipping translation")
                        translated_sft[field] = original
                        # Add success statistics (no translation needed)
                        if hasattr(translator, '_stats'):
                            add_translation_stats(translator._stats, f"sft_{field}", True)
                        continue
                    
                    translated = translator.translate_text(original)
                    
                    # Debug logging
                    logger.debug(f"Translation attempt for field '{field}':")
                    logger.debug(f"  Original: '{original[:50]}...'")
                    logger.debug(f"  Translated: '{translated[:50]}...'")
                    logger.debug(f"  Are they the same? {original == translated}")
                    
                    # Validate and sanitize translated field
                    if _validate_vi_translation(original, translated):
                        translated_sft[field] = _vi_sanitize_text(translated)
                        logger.debug(f"✅ Successfully translated field '{field}'")
                        # Add success statistics if stats available
                        if hasattr(translator, '_stats'):
                            add_translation_stats(translator._stats, f"sft_{field}", True)
                    else:
                        logger.warning(f"❌ Invalid Vietnamese translation for field {field}, keeping original")
                        logger.warning(f"  Original: '{original[:50]}...'")
                        logger.warning(f"  Translated: '{translated[:50]}...'")
                        translated_sft[field] = original
                        # Add failure statistics if stats available
                        if hasattr(translator, '_stats'):
                            add_translation_stats(translator._stats, f"sft_{field}", False)
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
        # Create a copy of the row to avoid modifying the original
        translated_row = row.copy()
        
        # Translate each field individually with proper validation
        for field in text_fields:
            if field in row and isinstance(row[field], str) and row[field].strip():
                try:
                    original = row[field]
                    
                    # Check if text is already in Vietnamese - skip translation if so
                    if _is_vietnamese_text(original):
                        logger.debug(f"RAG Field '{field}' is already in Vietnamese, skipping translation")
                        translated_row[field] = original
                        # Add success statistics (no translation needed)
                        if hasattr(translator, '_stats'):
                            add_translation_stats(translator._stats, f"rag_{field}", True)
                        continue
                    
                    translated = translator.translate_text(original)
                    
                    # Debug logging
                    logger.debug(f"RAG Translation attempt for field '{field}':")
                    logger.debug(f"  Original: '{original[:50]}...'")
                    logger.debug(f"  Translated: '{translated[:50]}...'")
                    logger.debug(f"  Are they the same? {original == translated}")
                    
                    # Validate and sanitize translated field
                    if _validate_vi_translation(original, translated):
                        translated_row[field] = _vi_sanitize_text(translated)
                        logger.debug(f"✅ Successfully translated RAG field '{field}'")
                        # Add success statistics if stats available
                        if hasattr(translator, '_stats'):
                            add_translation_stats(translator._stats, f"rag_{field}", True)
                    else:
                        logger.warning(f"❌ Invalid Vietnamese translation for RAG field {field}, keeping original")
                        logger.warning(f"  Original: '{original[:50]}...'")
                        logger.warning(f"  Translated: '{translated[:50]}...'")
                        translated_row[field] = original
                        # Add failure statistics if stats available
                        if hasattr(translator, '_stats'):
                            add_translation_stats(translator._stats, f"rag_{field}", False)
                except Exception as e:
                    logger.error(f"Failed to translate RAG field '{field}': {e}")
                    translated_row[field] = row[field]
            else:
                # Keep original if field doesn't exist or is empty
                translated_row[field] = row.get(field, "")
        
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
    
    if not translator:
        logger.warning("Vietnamese translation requested but translator is None")
        return False
    
    if not hasattr(translator, 'is_loaded') or not translator.is_loaded():
        logger.warning("Vietnamese translation requested but translator not loaded")
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

def add_translation_stats(stats: Dict[str, Any], field: str, success: bool) -> None:
    """
    Add translation statistics for individual fields.
    
    Args:
        stats: Statistics dictionary to update
        field: Field name that was translated
        success: Whether translation was successful
    """
    if "translation_stats" not in stats:
        stats["translation_stats"] = {}
    
    if field not in stats["translation_stats"]:
        stats["translation_stats"][field] = {"success": 0, "failed": 0}
    
    if success:
        stats["translation_stats"][field]["success"] += 1
    else:
        stats["translation_stats"][field]["failed"] += 1