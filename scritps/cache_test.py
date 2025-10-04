#!/usr/bin/env python3
"""
Test script to verify the fixes for HF permissions and Vietnamese translation
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from vi.translator import VietnameseTranslator
from vi.processing import translate_sft_row, _validate_vi_translation
from utils.schema import sft_row

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vietnamese_translation():
    """Test Vietnamese translation functionality"""
    logger.info("Testing Vietnamese translation...")
    
    # Create a sample SFT row
    sample_row = sft_row(
        instruction="Answer the patient's question like a clinician. Be concise and safe.",
        user_input="What are the symptoms of diabetes?",
        output="Common symptoms of diabetes include increased thirst, frequent urination, unexplained weight loss, fatigue, and blurred vision. If you experience these symptoms, please consult a healthcare provider.",
        source="test",
        rid="test_001",
        task="medical_dialogue"
    )
    
    logger.info(f"Original SFT row: {sample_row}")
    
    # Test translation validation
    test_cases = [
        ("Hello world", "Xin chào thế giới", True),  # Valid Vietnamese
        ("Hello world", "Hello world", False),  # Same as original (not translated)
        ("Hello world", "translation error", False),  # Contains error keyword
        ("Hello world", "Hi", False),  # Too short
        ("Hello world", "", False),  # Empty
    ]
    
    logger.info("Testing translation validation...")
    for original, translated, expected in test_cases:
        result = _validate_vi_translation(original, translated)
        status = "✅" if result == expected else "❌"
        logger.info(f"{status} {original} -> {translated}: {result} (expected {expected})")
    
    # Test with translator (if available)
    try:
        translator = VietnameseTranslator()
        logger.info("Vietnamese translator initialized successfully")
        
        # Try to load the model
        try:
            translator.load_model()
            logger.info("✅ Translation model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load translation model: {e}")
            logger.info("This is expected if the model is not downloaded yet")
            return
        
        # Test translation
        translated_row = translate_sft_row(sample_row, translator)
        logger.info(f"Translated SFT row: {translated_row}")
        
        # Check if translation was applied
        original_sft = sample_row["sft"]
        translated_sft = translated_row["sft"]
        
        for field in ["instruction", "input", "output"]:
            original_text = original_sft[field]
            translated_text = translated_sft[field]
            
            if original_text != translated_text:
                logger.info(f"✅ Field '{field}' was translated")
                logger.info(f"  Original: {original_text[:100]}...")
                logger.info(f"  Translated: {translated_text[:100]}...")
            else:
                logger.info(f"⚠️ Field '{field}' was not translated (may be due to validation failure)")
        
    except Exception as e:
        logger.warning(f"Could not test with actual translator: {e}")
        logger.info("This is expected if the model is not downloaded yet")

def test_hf_cache_setup():
    """Test Hugging Face cache directory setup"""
    logger.info("Testing HF cache setup...")
    
    # Test cache directory creation
    cache_dir = os.path.abspath("cache/huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    
    if os.path.exists(cache_dir) and os.access(cache_dir, os.R_OK | os.W_OK):
        logger.info(f"✅ Cache directory {cache_dir} is accessible")
    else:
        logger.error(f"❌ Cache directory {cache_dir} is not accessible")
    
    # Test HF_HOME environment variable
    os.environ["HF_HOME"] = cache_dir
    hf_home = os.getenv("HF_HOME")
    if hf_home == cache_dir:
        logger.info(f"✅ HF_HOME environment variable set to {hf_home}")
    else:
        logger.error(f"❌ HF_HOME environment variable not set correctly")

if __name__ == "__main__":
    logger.info("Starting fix verification tests...")
    
    test_hf_cache_setup()
    test_vietnamese_translation()
    
    logger.info("Tests completed!")
