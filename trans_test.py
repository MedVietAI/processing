#!/usr/bin/env python3
"""
Test script for Vietnamese translation functionality
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vi.translator import VietnameseTranslator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_translation():
    """Test the Vietnamese translation functionality"""
    load_dotenv()
    
    # Initialize translator
    translator = VietnameseTranslator()
    
    try:
        # Load the model
        logger.info("Loading translation model...")
        translator.load_model()
        logger.info("✅ Model loaded successfully")
        
        # Test single text translation
        test_text = "Hello, how are you today? I hope you are feeling well."
        logger.info(f"Original text: {test_text}")
        
        translated = translator.translate_text(test_text)
        logger.info(f"Translated text: {translated}")
        
        # Test batch translation
        test_texts = [
            "What are the symptoms of diabetes?",
            "How do I treat a headache?",
            "What is the recommended dosage for this medication?"
        ]
        
        logger.info("Testing batch translation...")
        batch_translated = translator.translate_batch(test_texts)
        
        for i, (original, translated) in enumerate(zip(test_texts, batch_translated)):
            logger.info(f"Batch {i+1}:")
            logger.info(f"  Original: {original}")
            logger.info(f"  Translated: {translated}")
        
        # Test dictionary translation
        test_dict = {
            "instruction": "Answer the medical question",
            "input": "What are the side effects of aspirin?",
            "output": "Common side effects include stomach irritation and bleeding."
        }
        
        logger.info("Testing dictionary translation...")
        dict_translated = translator.translate_dict(test_dict, ["instruction", "input", "output"])
        
        logger.info("Dictionary translation result:")
        for key, value in dict_translated.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("🎉 All translation tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Translation test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_translation()
    sys.exit(0 if success else 1)
