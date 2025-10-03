"""
Model Download Script for Vietnamese Translation

This script downloads the Helsinki-NLP/opus-mt-en-vi model
and saves it to the Hugging Face cache directory.
"""

import os
import sys
import logging
from pathlib import Path
import torch
from transformers import MarianMTModel, MarianTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_model(model_name: str = "Helsinki-NLP/opus-mt-en-vi", cache_dir: str = None):
    """
    Download the translation model and tokenizer.
    
    Args:
        model_name: Hugging Face model name
        cache_dir: Cache directory for the model. If None, uses HF_HOME env var
    """
    if cache_dir is None:
        cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    
    logger.info(f"Downloading model: {model_name}")
    logger.info(f"Cache directory: {cache_dir}")
    
    try:
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = MarianTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        logger.info("✅ Tokenizer downloaded successfully")
        
        # Download model
        logger.info("Downloading model...")
        model = MarianMTModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        logger.info("✅ Model downloaded successfully")
        
        # Test the model
        logger.info("Testing model...")
        test_text = "Hello, how are you?"
        inputs = tokenizer(f">>vie<< {test_text}", return_tensors="pt")
        model.eval()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50, num_beams=4)
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Test translation: '{test_text}' -> '{translated}'")
        
        logger.info("🎉 Model download and test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download model: {e}")
        return False

def main():
    """Main function to download the model."""
    # Get model name from environment variable or use default
    model_name = os.getenv("EN_VI", "Helsinki-NLP/opus-mt-en-vi")
    
    logger.info("Starting model download process...")
    logger.info(f"Model: {model_name}")
    
    success = download_model(model_name)
    
    if success:
        logger.info("Model download completed successfully!")
        sys.exit(0)
    else:
        logger.error("Model download failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
