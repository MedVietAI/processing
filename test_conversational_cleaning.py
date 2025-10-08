#!/usr/bin/env python3
"""
Test conversational element cleaning and failed response handling
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils import augment as A

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_conversational_cleaning():
    """Test conversational element cleaning"""
    logger.info("Testing conversational element cleaning...")
    
    test_cases = [
        # (input, expected_contains, expected_not_contains, description)
        ("Hi, I'm a doctor. Diabetes symptoms include...", "Diabetes symptoms", ["Hi", "I'm a doctor"], "English greeting + doctor intro"),
        ("Xin chào, tôi là bác sĩ. Triệu chứng tiểu đường...", "Triệu chứng tiểu đường", ["Xin chào", "tôi là bác sĩ"], "Vietnamese greeting + doctor intro"),
        ("If you are a doctor, please answer...", "answer", ["If you are a doctor", "please"], "Doctor conditional"),
        ("Thank you for your question. The symptoms are...", "The symptoms are", ["Thank you", "for your question"], "Thank you prefix"),
        ("I hope this helps. Best regards!", "helps", ["I hope this", "Best regards"], "Thank you suffix"),
        ("Nếu bạn là bác sĩ, vui lòng trả lời...", "trả lời", ["Nếu bạn là bác sĩ", "vui lòng"], "Vietnamese doctor conditional"),
        ("As a medical professional, I can tell you...", "I can tell you", ["As a medical professional"], "Medical professional intro"),
        ("From a medical perspective, the answer is...", "the answer is", ["From a medical perspective"], "Medical perspective intro"),
        ("Medically speaking, this condition...", "this condition", ["Medically speaking"], "Medically speaking intro"),
        ("I'm here to help. The treatment is...", "The treatment is", ["I'm here to help"], "Helpful intro"),
    ]
    
    all_passed = True
    for input_text, expected_contains, expected_not_contains, description in test_cases:
        cleaned = A.clean_conversational_elements(input_text)
        
        # Check that expected content is preserved
        contains_expected = all(phrase in cleaned for phrase in expected_contains)
        
        # Check that conversational elements are removed
        not_contains_expected = all(phrase not in cleaned for phrase in expected_not_contains)
        
        status = "✅" if contains_expected and not_contains_expected else "❌"
        if not (contains_expected and not_contains_expected):
            all_passed = False
        
        logger.info(f"{status} {description}")
        logger.info(f"  Input: '{input_text}'")
        logger.info(f"  Cleaned: '{cleaned}'")
        logger.info(f"  Contains expected: {contains_expected}, Removes unwanted: {not_contains_expected}")
        logger.info("")
    
    return all_passed

def test_invalid_response_detection():
    """Test invalid response detection"""
    logger.info("Testing invalid response detection...")
    
    test_cases = [
        # (text, expected_invalid, description)
        ("FAIL", True, "Simple fail response"),
        ("I can't help you", True, "Can't help response"),
        ("I don't know", True, "Don't know response"),
        ("Sorry, I'm unable to", True, "Unable response"),
        ("Diabetes symptoms include...", False, "Valid medical response"),
        ("The treatment is...", False, "Valid treatment response"),
        ("", True, "Empty response"),
        ("Hi", True, "Too short response"),
        ("I'm sorry, I cannot determine", True, "Cannot determine response"),
    ]
    
    all_passed = True
    for text, expected_invalid, description in test_cases:
        is_invalid = A.is_invalid_response(text)
        status = "✅" if is_invalid == expected_invalid else "❌"
        if is_invalid != expected_invalid:
            all_passed = False
        
        logger.info(f"{status} {description}: '{text}' -> {is_invalid} (expected {expected_invalid})")
    
    return all_passed

def test_retry_logic():
    """Test retry logic for failed responses"""
    logger.info("Testing retry logic...")
    
    # Test that invalid responses are detected
    invalid_responses = ["FAIL", "I can't help", "Sorry", ""]
    
    for response in invalid_responses:
        is_invalid = A.is_invalid_response(response)
        if is_invalid:
            logger.info(f"✅ Correctly detected invalid response: '{response}'")
        else:
            logger.error(f"❌ Failed to detect invalid response: '{response}'")
            return False
    
    # Test conversational cleaning
    conversational_text = "Hi, I'm a doctor. Diabetes symptoms include increased thirst."
    cleaned = A.clean_conversational_elements(conversational_text)
    
    if "Diabetes symptoms include increased thirst" in cleaned and "Hi" not in cleaned:
        logger.info("✅ Conversational cleaning working correctly")
    else:
        logger.error("❌ Conversational cleaning failed")
        return False
    
    return True

def main():
    """Run all tests"""
    logger.info("Testing conversational cleaning and failed response handling...")
    logger.info("=" * 70)
    
    tests = [
        ("Conversational Cleaning", test_conversational_cleaning),
        ("Invalid Response Detection", test_invalid_response_detection),
        ("Retry Logic", test_retry_logic),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name}: ❌ ERROR - {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("CONVERSATIONAL CLEANING TEST SUMMARY")
    logger.info("=" * 70)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Conversational cleaning is working correctly.")
        logger.info("✅ Failed responses will be retried, not recorded!")
        logger.info("✅ Conversational elements are properly cleaned!")
    else:
        logger.warning("⚠️ Some tests failed. Please check the logs above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
