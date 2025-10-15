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
            
            # Generate with optimized parameters for MedAlpaca
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    top_p=0.9 if temperature > 0 else 1.0,
                    top_k=50 if temperature > 0 else 0,
                    num_beams=1 if temperature > 0 else 4,
                    early_stopping=True
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
        """Format prompt for MedAlpaca model with medical-specific formatting"""
        # MedAlpaca was trained on medical Q&A pairs, so we use its expected format
        if "Question:" in prompt and "Answer:" in prompt:
            return prompt
        elif "Context:" in prompt and "Question:" in prompt:
            return prompt
        elif "You are a" in prompt or "medical" in prompt.lower():
            # For medical instructions, use Alpaca format
            return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:"
        else:
            # Default medical Q&A format for MedAlpaca
            return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nAnswer the following medical question accurately and professionally.\n\n### Input:\n{prompt}\n\n### Response:"
    
    def _clean_response(self, text: str) -> str:
        """Clean generated response with medical-specific cleaning"""
        if not text:
            return text
            
        # Remove common conversational prefixes and comments
        prefixes_to_remove = [
            "Answer:",
            "The answer is:",
            "Based on the information provided:",
            "Here's the answer:",
            "Here is the answer:",
            "Here's a rewritten version:",
            "Here is a rewritten version:",
            "Here's the rewritten text:",
            "Here is the rewritten text:",
            "Here's the translation:",
            "Here is the translation:",
            "Here's the enhanced text:",
            "Here is the enhanced text:",
            "Here's the improved text:",
            "Here is the improved text:",
            "Here's the medical context:",
            "Here is the medical context:",
            "Here's the cleaned text:",
            "Here is the cleaned text:",
            "Sure,",
            "Okay,",
            "Certainly,",
            "Of course,",
            "I can help you with that.",
            "I'll help you with that.",
            "Let me help you with that.",
            "I can rewrite that for you.",
            "I'll rewrite that for you.",
            "Let me rewrite that for you.",
            "I can translate that for you.",
            "I'll translate that for you.",
            "Let me translate that for you.",
            "### Response:",
            "Response:",
            "Below is an instruction",
            "### Instruction:",
            "Instruction:",
        ]
        
        text = text.strip()
        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
                break
        
        # Remove any remaining Alpaca format artifacts
        if "### Response:" in text:
            text = text.split("### Response:")[-1].strip()
        if "### Input:" in text:
            text = text.split("### Input:")[0].strip()
        
        # Remove any remaining conversational elements
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not any(phrase in line.lower() for phrase in [
                "here's", "here is", "let me", "i can", "i'll", "sure,", "okay,", 
                "certainly,", "of course,", "i hope this helps", "hope this helps",
                "does this help", "is this what you", "let me know if"
            ]):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _snip(self, text: str, max_words: int = 12) -> str:
        """Truncate text for logging"""
        if not text:
            return "∅"
        words = text.strip().split()
        return " ".join(words[:max_words]) + (" …" if len(words) > max_words else "")
    
    def generate_batch(self, prompts: list, max_tokens: int = 512, temperature: float = 0.2) -> list:
        """Generate text for multiple prompts in batch for better efficiency"""
        if not self.is_loaded:
            self.load_model()
            
        if not prompts:
            return []
            
        try:
            # Format all prompts
            formatted_prompts = [self._format_prompt(prompt) for prompt in prompts]
            
            # Tokenize all inputs
            inputs = self.tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate for all prompts
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    top_p=0.9 if temperature > 0 else 1.0,
                    top_k=50 if temperature > 0 else 0,
                    num_beams=1 if temperature > 0 else 4,
                    early_stopping=True
                )
            
            # Decode all outputs
            results = []
            input_length = inputs['input_ids'].shape[1]
            for i, output in enumerate(outputs):
                generated_text = self.tokenizer.decode(
                    output[input_length:],
                    skip_special_tokens=True
                ).strip()
                cleaned_text = self._clean_response(generated_text)
                results.append(cleaned_text)
            
            logger.info(f"[LOCAL_LLM] Generated batch of {len(prompts)} texts")
            return results
            
        except Exception as e:
            logger.error(f"[LOCAL_LLM] Batch generation failed: {e}")
            return [None] * len(prompts)
    
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
        """Paraphrase text using MedAlpaca with medical-specific optimization"""
        if not text or len(text) < 12:
            return text
            
        if custom_prompt:
            prompt = custom_prompt
        else:
            # Medical-specific paraphrasing prompts based on difficulty
            if difficulty == "easy":
                prompt = (
                    "Rewrite the following medical text using different words while preserving all medical facts, clinical terms, and meaning. Keep the same level of detail and accuracy. Return only the rewritten text without any introduction or commentary.\n\n"
                    f"{text}"
                )
            else:  # hard difficulty
                prompt = (
                    "Rewrite the following medical text using more sophisticated medical language and different sentence structures while preserving all clinical facts, medical terminology, and diagnostic information. Maintain professional medical tone. Return only the rewritten text without any introduction or commentary.\n\n"
                    f"{text}"
                )
        
        # Adjust temperature based on difficulty
        temperature = 0.1 if difficulty == "easy" else 0.3
        result = self.client.generate(prompt, max_tokens=min(600, max(128, len(text)//2)), temperature=temperature)
        return result if result else text
    
    def translate(self, text: str, target_lang: str = "vi") -> Optional[str]:
        """Translate text using MedAlpaca with medical terminology preservation"""
        if not text:
            return text
            
        # Medical-specific translation prompt
        if target_lang == "vi":
            prompt = (
                "Translate the following English medical text to Vietnamese while preserving all medical terminology, clinical facts, and professional medical language. Use appropriate Vietnamese medical terms. Return only the translation without any introduction or commentary.\n\n"
                f"{text}"
            )
        else:
            prompt = (
                f"Translate the following medical text to {target_lang} while preserving all medical terminology, clinical facts, and professional medical language. Return only the translation without any introduction or commentary.\n\n"
                f"{text}"
            )
        
        result = self.client.generate(prompt, max_tokens=min(800, len(text)+100), temperature=0.0)
        return result.strip() if result else None
    
    def backtranslate(self, text: str, via_lang: str = "vi") -> Optional[str]:
        """Backtranslate text using MedAlpaca with medical accuracy"""
        if not text:
            return text
            
        # First translate to target language
        translated = self.translate(text, target_lang=via_lang)
        if not translated:
            return None
            
        # Then translate back to English with medical focus
        if via_lang == "vi":
            prompt = (
                "Translate the following Vietnamese medical text back to English while preserving all medical terminology, clinical facts, and professional medical language. Ensure the translation is medically accurate. Return only the translation without any introduction or commentary.\n\n"
                f"{translated}"
            )
        else:
            prompt = (
                f"Translate the following {via_lang} medical text back to English while preserving all medical terminology, clinical facts, and professional medical language. Return only the translation without any introduction or commentary.\n\n"
                f"{translated}"
            )
        
        result = self.client.generate(prompt, max_tokens=min(900, len(text)+150), temperature=0.0)
        return result.strip() if result else None
    
    def consistency_check(self, user: str, output: str) -> bool:
        """Check consistency using MedAlpaca with medical validation focus"""
        prompt = (
            "Evaluate if the medical answer is consistent with the question/context and medically accurate. Consider medical accuracy, clinical appropriateness, consistency with the question, safety standards, and completeness of medical information. Reply with exactly 'PASS' if the answer is medically sound and consistent, otherwise 'FAIL'.\n\n"
            f"Question/Context: {user}\n\n"
            f"Medical Answer: {output}"
        )
        
        result = self.client.generate(prompt, max_tokens=5, temperature=0.0)
        return isinstance(result, str) and "PASS" in result.upper()
    
    def medical_accuracy_check(self, question: str, answer: str) -> bool:
        """Check medical accuracy of Q&A pairs using MedAlpaca"""
        if not question or not answer:
            return False
            
        prompt = (
            "Evaluate if the medical answer is accurate and appropriate for the question. Consider medical facts, clinical knowledge, appropriate medical terminology, clinical reasoning, logic, and safety considerations. Reply with exactly 'ACCURATE' if the answer is medically correct, otherwise 'INACCURATE'.\n\n"
            f"Medical Question: {question}\n\n"
            f"Medical Answer: {answer}"
        )
        
        result = self.client.generate(prompt, max_tokens=5, temperature=0.0)
        return isinstance(result, str) and "ACCURATE" in result.upper()
    
    def enhance_medical_terminology(self, text: str) -> str:
        """Enhance medical terminology in text using MedAlpaca"""
        if not text or len(text) < 20:
            return text
            
        prompt = (
            "Improve the medical terminology in the following text while preserving all factual information and clinical accuracy. Use more precise medical terms where appropriate. Return only the improved text without any introduction or commentary.\n\n"
            f"{text}"
        )
        
        result = self.client.generate(prompt, max_tokens=min(800, len(text)+100), temperature=0.1)
        return result if result else text
    
    def create_clinical_scenarios(self, question: str, answer: str) -> list:
        """Create different clinical scenarios from Q&A pairs using MedAlpaca with batch optimization"""
        scenarios = []
        
        # Different clinical context prompts
        context_prompts = [
            (
                "Rewrite this medical question as if asked by a patient in an emergency room setting. Return only the rewritten question without any introduction or commentary:\n\n{question}",
                "emergency_room"
            ),
            (
                "Rewrite this medical question as if asked by a patient during a routine checkup. Return only the rewritten question without any introduction or commentary:\n\n{question}",
                "routine_checkup"
            ),
            (
                "Rewrite this medical question as if asked by a patient with chronic conditions. Return only the rewritten question without any introduction or commentary:\n\n{question}",
                "chronic_care"
            ),
            (
                "Rewrite this medical question as if asked by a patient's family member. Return only the rewritten question without any introduction or commentary:\n\n{question}",
                "family_inquiry"
            )
        ]
        
        # Use batch processing for better efficiency
        try:
            prompts = [prompt_template.format(question=question) for prompt_template, _ in context_prompts]
            results = self.client.generate_batch(prompts, max_tokens=min(400, len(question)+50), temperature=0.2)
            
            for i, (result, (_, scenario_type)) in enumerate(zip(results, context_prompts)):
                if result and not self._is_invalid_response(result):
                    scenarios.append((result, answer, scenario_type))
                    
        except Exception as e:
            logger.warning(f"Batch clinical scenario creation failed, falling back to individual: {e}")
            # Fallback to individual processing
            for prompt_template, scenario_type in context_prompts:
                try:
                    prompt = prompt_template.format(question=question)
                    scenario_question = self.client.generate(prompt, max_tokens=min(400, len(question)+50), temperature=0.2)
                    
                    if scenario_question and not self._is_invalid_response(scenario_question):
                        scenarios.append((scenario_question, answer, scenario_type))
                except Exception as e:
                    logger.warning(f"Failed to create clinical scenario {scenario_type}: {e}")
                    continue
                
        return scenarios
    
    def _is_invalid_response(self, text: str) -> bool:
        """Check if response is invalid (similar to augment.py)"""
        if not text or not isinstance(text, str):
            return True
        
        text_lower = text.lower().strip()
        invalid_patterns = [
            "fail", "invalid", "i couldn't", "i can't", "i cannot", "unable to",
            "sorry", "error", "not available", "no answer", "insufficient",
            "don't know", "do not know", "not sure", "cannot determine",
            "unable to provide", "not possible", "not applicable", "n/a"
        ]
        
        if len(text_lower) < 3:
            return True
        
        for pattern in invalid_patterns:
            if pattern in text_lower:
                return True
        
        return False
    
    def unload(self):
        """Unload the model"""
        self.client.unload_model()
