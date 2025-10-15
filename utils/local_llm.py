# Local MedAlpaca-13b inference client
import os
import logging
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc

logger = logging.getLogger("local_llm")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

class MedAlpacaClient:
    """Local MedAlpaca-13b client for medical text generation"""
    
    def __init__(self, model_name: str = "medalpaca/medalpaca-13b", hf_token: str = None):
        self.model_name = model_name
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        
        logger.info(f"[LOCAL_LLM] Initializing MedAlpaca client on device: {self.device}")
        
    def load_model(self):
        """Load the MedAlpaca model and tokenizer"""
        if self.is_loaded:
            return
            
        try:
            logger.info(f"[LOCAL_LLM] Loading MedAlpaca model: {self.model_name}")
            
            # Configure quantization for memory efficiency
            if self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                quantization_config = None
                
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token,
                cache_dir=os.getenv("HF_HOME", "~/.cache/huggingface")
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.hf_token,
                cache_dir=os.getenv("HF_HOME", "~/.cache/huggingface"),
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            self.is_loaded = True
            logger.info("[LOCAL_LLM] MedAlpaca model loaded successfully")
            
        except Exception as e:
            logger.error(f"[LOCAL_LLM] Failed to load model: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> Optional[str]:
        """Generate text using MedAlpaca model"""
        if not self.is_loaded:
            self.load_model()
            
        try:
            # Format prompt for MedAlpaca
            formatted_prompt = self._format_prompt(prompt)
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # Clean up response
            cleaned_text = self._clean_response(generated_text)
            
            logger.info(f"[LOCAL_LLM] Generated: {self._snip(cleaned_text)}")
            return cleaned_text
            
        except Exception as e:
            logger.error(f"[LOCAL_LLM] Generation failed: {e}")
            return None
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt for MedAlpaca model"""
        # MedAlpaca uses a specific format for medical Q&A
        if "Question:" in prompt and "Answer:" in prompt:
            return prompt
        elif "Context:" in prompt and "Question:" in prompt:
            return prompt
        else:
            # Simple medical Q&A format
            return f"Question: {prompt}\n\nAnswer:"
    
    def _clean_response(self, text: str) -> str:
        """Clean generated response"""
        if not text:
            return text
            
        # Remove common prefixes
        prefixes_to_remove = [
            "Answer:",
            "The answer is:",
            "Based on the information provided:",
            "Here's the answer:",
            "Here is the answer:",
        ]
        
        text = text.strip()
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                break
                
        return text
    
    def _snip(self, text: str, max_words: int = 12) -> str:
        """Truncate text for logging"""
        if not text:
            return "∅"
        words = text.strip().split()
        return " ".join(words[:max_words]) + (" …" if len(words) > max_words else "")
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        self.is_loaded = False
        logger.info("[LOCAL_LLM] Model unloaded and memory freed")

class LocalParaphraser:
    """Local paraphraser using MedAlpaca model"""
    
    def __init__(self, model_name: str = "medalpaca/medalpaca-13b", hf_token: str = None):
        self.client = MedAlpacaClient(model_name, hf_token)
        
    def paraphrase(self, text: str, difficulty: str = "easy", custom_prompt: str = None) -> str:
        """Paraphrase text using MedAlpaca"""
        if not text or len(text) < 12:
            return text
            
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = (
                "Paraphrase the following medical text concisely, preserve meaning and clinical terms.\n"
                "Do not fabricate or remove factual claims.\n" 
                "Return ONLY the rewritten text, without any introduction, commentary.\n\n"
                f"Original text: {text}"
            )
        
        result = self.client.generate(prompt, max_tokens=min(600, max(128, len(text)//2)), temperature=0.1)
        return result if result else text
    
    def translate(self, text: str, target_lang: str = "vi") -> Optional[str]:
        """Translate text using MedAlpaca"""
        if not text:
            return text
            
        prompt = f"Translate the following medical text to {target_lang}. Keep meaning exact, preserve medical terms:\n\n{text}"
        result = self.client.generate(prompt, max_tokens=min(800, len(text)+100), temperature=0.0)
        return result.strip() if result else None
    
    def backtranslate(self, text: str, via_lang: str = "vi") -> Optional[str]:
        """Backtranslate text using MedAlpaca"""
        if not text:
            return text
            
        # First translate to target language
        translated = self.translate(text, target_lang=via_lang)
        if not translated:
            return None
            
        # Then translate back to English
        prompt = f"Translate the following {via_lang} text back to English, preserving the exact meaning:\n\n{translated}"
        result = self.client.generate(prompt, max_tokens=min(900, len(text)+150), temperature=0.0)
        return result.strip() if result else None
    
    def consistency_check(self, user: str, output: str) -> bool:
        """Check consistency using MedAlpaca"""
        prompt = (
            "You are a strict medical QA validator. Given the USER input (question+context) "
            "and the MODEL ANSWER, reply with exactly 'PASS' if the answer is supported and safe, "
            "otherwise 'FAIL'. No extra text.\n\n"
            f"USER:\n{user}\n\nANSWER:\n{output}"
        )
        
        result = self.client.generate(prompt, max_tokens=3, temperature=0.0)
        return isinstance(result, str) and "PASS" in result.upper()
    
    def unload(self):
        """Unload the model"""
        self.client.unload_model()
