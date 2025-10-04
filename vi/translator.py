"""
Vietnamese Translator using Helsinki-NLP/opus-mt-en-vi model
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from transformers import MarianMTModel, MarianTokenizer
import torch

logger = logging.getLogger(__name__)

class VietnameseTranslator:
    """
    Vietnamese translator using Helsinki-NLP/opus-mt-en-vi model.
    
    This class handles translation from English to Vietnamese using the
    MarianMT model from Hugging Face Transformers.
    """
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the Vietnamese translator.
        
        Args:
            model_name: Hugging Face model name. Defaults to EN_VI env var or Helsinki-NLP/opus-mt-en-vi
            device: Device to run the model on ('cpu', 'cuda', 'auto'). Defaults to 'auto'
        """
        self.model_name = model_name or os.getenv("EN_VI", "Helsinki-NLP/opus-mt-en-vi")
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        
        logger.info(f"VietnameseTranslator initialized with model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
    
    def _get_device(self, device: Optional[str]) -> str:
        """Determine the best device to use for the model."""
        if device:
            return device
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_model(self) -> None:
        """Load the translation model and tokenizer."""
        if self._is_loaded:
            logger.debug("Model already loaded, skipping...")
            return
        
        try:
            logger.info(f"Loading translation model: {self.model_name}")
            logger.info(f"Loading on device: {self.device}")
            
            # Set up cache directory
            cache_dir = os.getenv("HF_HOME", os.path.abspath("cache/huggingface"))
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load tokenizer
            self.tokenizer = MarianTokenizer.from_pretrained(
                self.model_name,
                cache_dir=cache_dir
            )
            
            # Load model
            self.model = MarianMTModel.from_pretrained(
                self.model_name,
                cache_dir=cache_dir
            )
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self._is_loaded = True
            logger.info("✅ Translation model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load translation model: {e}")
            raise RuntimeError(f"Failed to load translation model: {e}")
    
    def translate_text(self, text: str) -> str:
        """
        Translate a single text from English to Vietnamese.
        
        Args:
            text: English text to translate
            
        Returns:
            Translated Vietnamese text
        """
        if not self._is_loaded:
            self.load_model()
        
        if not text or not text.strip():
            return text
        
        try:
            # Prepare input with target language token
            # The model requires a target language token in the format >>id<<
            input_text = f">>vie<< {text.strip()}"
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Translate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode
            translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            logger.debug(f"Translated: '{text[:50]}...' -> '{translated[:50]}...'")
            return translated.strip()
            
        except Exception as e:
            logger.error(f"Translation failed for text: '{text[:100]}...' - Error: {e}")
            # Return original text if translation fails
            return text
    
    def translate_batch(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """
        Translate a batch of texts from English to Vietnamese.
        
        Args:
            texts: List of English texts to translate
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of translated Vietnamese texts
        """
        if not self._is_loaded:
            self.load_model()
        
        if not texts:
            return []
        
        results = []
        
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
                # Prepare batch with target language tokens
                batch_inputs = [f">>vie<< {text.strip()}" for text in batch]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Translate batch
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=512,
                        num_beams=4,
                        early_stopping=True,
                        do_sample=False
                    )
                
                # Decode batch
                batch_translations = [
                    self.tokenizer.decode(output, skip_special_tokens=True).strip()
                    for output in outputs
                ]
                
                results.extend(batch_translations)
                
        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            # Return original texts if translation fails
            results = texts
        
        logger.info(f"Translated {len(texts)} texts successfully")
        return results
    
    def translate_dict(self, data: Dict[str, Any], text_fields: List[str]) -> Dict[str, Any]:
        """
        Translate specific text fields in a dictionary from English to Vietnamese.
        
        Args:
            data: Dictionary containing the data
            text_fields: List of field names to translate
            
        Returns:
            Dictionary with translated text fields
        """
        if not self._is_loaded:
            self.load_model()
        
        result = data.copy()
        
        for field in text_fields:
            if field in data and isinstance(data[field], str) and data[field].strip():
                try:
                    result[field] = self.translate_text(data[field])
                    logger.debug(f"Translated field '{field}': '{data[field][:50]}...' -> '{result[field][:50]}...'")
                except Exception as e:
                    logger.error(f"Failed to translate field '{field}': {e}")
                    # Keep original text if translation fails
                    result[field] = data[field]
        
        return result
    
    def translate_list_of_dicts(self, data_list: List[Dict[str, Any]], text_fields: List[str]) -> List[Dict[str, Any]]:
        """
        Translate specific text fields in a list of dictionaries.
        
        Args:
            data_list: List of dictionaries containing the data
            text_fields: List of field names to translate in each dictionary
            
        Returns:
            List of dictionaries with translated text fields
        """
        if not data_list:
            return []
        
        logger.info(f"Translating {len(data_list)} items with fields: {text_fields}")
        
        results = []
        for i, data in enumerate(data_list):
            try:
                translated_data = self.translate_dict(data, text_fields)
                results.append(translated_data)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Translated {i + 1}/{len(data_list)} items")
                    
            except Exception as e:
                logger.error(f"Failed to translate item {i}: {e}")
                results.append(data)  # Keep original data if translation fails
        
        logger.info(f"Completed translation of {len(data_list)} items")
        return results
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "is_loaded": self._is_loaded
        }
