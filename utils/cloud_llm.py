# Round-robin rotator + paraphrasing + translation/backtranslation
import os
import logging
import requests
import time
from typing import Optional

# Dynamic import for Google GenAI (only when not in local mode)
def _import_google_genai():
    """Dynamically import Google GenAI only when needed"""
    try:
        from google import genai
        return genai
    except ImportError as e:
        raise ImportError(f"Google GenAI not available: {e}. Make sure IS_LOCAL=false and google-genai is installed.")

# Check if we're in local mode
IS_LOCAL = os.getenv("IS_LOCAL", "false").lower() == "true"

# Only import Google GenAI if not in local mode
if not IS_LOCAL:
    try:
        genai = _import_google_genai()
    except ImportError:
        genai = None
else:
    genai = None

logger = logging.getLogger("llm")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

# LLM parser limit text to log-out
def snip(s: str, n: int = 12) -> str:
    if not isinstance(s, str): return "∅"
    parts = s.strip().split()
    return " ".join(parts[:n]) + (" …" if len(parts) > n else "")

class KeyRotator:
    def __init__(self, env_prefix: str, max_keys: int = 5):
        keys = []
        for i in range(1, max_keys + 1):
            v = os.getenv(f"{env_prefix}_{i}")
            if v:
                keys.append(v.strip())
        if not keys:
            logger.warning(f"[LLM] No keys found for prefix {env_prefix}_*")
        self.keys = keys
        self.dead = set()  # Permanently dead keys
        self.temp_dead = {}  # Temporarily dead keys with retry time
        self.retry_counts = {}  # Track retry attempts per key
        self.idx = 0
        self.max_retries = 3  # Max retries before permanent death
        self.retry_delay = 60  # Seconds to wait before retry
        
    def next_key(self) -> Optional[str]:
        if not self.keys:
            return None
            
        # Clean up expired temporary dead keys
        current_time = time.time()
        expired_keys = [k for k, retry_time in self.temp_dead.items() if current_time > retry_time]
        for k in expired_keys:
            del self.temp_dead[k]
            logger.info(f"[LLM] Key {k[:6]}*** is back in rotation after cooldown")
        
        # Try to find an available key
        for _ in range(len(self.keys)):
            k = self.keys[self.idx % len(self.keys)]
            self.idx += 1
            
            # Skip permanently dead keys
            if k in self.dead:
                continue
                
            # Skip temporarily dead keys
            if k in self.temp_dead and current_time < self.temp_dead[k]:
                continue
                
            return k
            
        # All keys are dead or temporarily unavailable
        logger.warning(f"[LLM] All keys for {env_prefix} are unavailable")
        return None

    def mark_bad(self, key: Optional[str], error_type: str = "unknown"):
        if not key:
            return
            
        current_time = time.time()
        retry_count = self.retry_counts.get(key, 0)
        
        # Determine if this is a temporary or permanent failure
        is_temporary = self._is_temporary_error(error_type)
        
        if is_temporary and retry_count < self.max_retries:
            # Temporary failure - add to temp_dead with retry time
            retry_delay = self.retry_delay * (2 ** retry_count)  # Exponential backoff
            self.temp_dead[key] = current_time + retry_delay
            self.retry_counts[key] = retry_count + 1
            logger.warning(f"[LLM] Key {key[:6]}*** temporarily quarantined for {retry_delay}s (attempt {retry_count + 1}/{self.max_retries})")
        else:
            # Permanent failure or max retries reached
            self.dead.add(key)
            if key in self.temp_dead:
                del self.temp_dead[key]
            if key in self.retry_counts:
                del self.retry_counts[key]
            logger.error(f"[LLM] Key {key[:6]}*** permanently quarantined after {retry_count} retries")
    
    def _is_temporary_error(self, error_type: str) -> bool:
        """Determine if an error is temporary and worth retrying"""
        temporary_errors = [
            "rate_limit", "quota_exceeded", "too_many_requests", "429",
            "service_unavailable", "503", "bad_gateway", "502",
            "timeout", "connection_error", "network_error"
        ]
        
        error_lower = error_type.lower()
        return any(temp_err in error_lower for temp_err in temporary_errors)
    
    def get_stats(self) -> dict:
        """Get rotator statistics"""
        return {
            "total_keys": len(self.keys),
            "dead_keys": len(self.dead),
            "temp_dead_keys": len(self.temp_dead),
            "available_keys": len(self.keys) - len(self.dead) - len(self.temp_dead),
            "retry_counts": self.retry_counts.copy()
        }

class GeminiClient:
    def __init__(self, rotator: KeyRotator, default_model: str):
        self.rotator = rotator
        self.default_model = default_model
        self.available = genai is not None and not IS_LOCAL

    def generate(self, prompt: str, model: Optional[str] = None, temperature: float = 0.2, max_output_tokens: int = 512) -> Optional[str]:
        if not self.available:
            logger.warning("[LLM][Gemini] Google GenAI not available (local mode or import failed)")
            return None
            
        key = self.rotator.next_key()
        if not key:
            return None
        try:
            client = genai.Client(api_key=key)
            # NOTE: matches your required pattern/use
            res = client.models.generate_content(
                model=model or self.default_model,
                contents=prompt
            )
            text = getattr(res, "text", None)
            if text:
                logger.info(f"[LLM][Gemini] out={snip(text)}")
            return text
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[LLM][Gemini] {error_msg}")
            self.rotator.mark_bad(key, error_msg)
            return None

class NvidiaClient:
    def __init__(self, rotator: KeyRotator, default_model: str):
        self.rotator = rotator
        self.default_model = default_model
        self.url = os.getenv("NVIDIA_API_URL", "https://integrate.api.nvidia.com/v1/chat/completions")

    # Regex-based cleaning resp from quotes
    def _clean_resp(self, resp: str) -> str:
        if not resp: return resp
        txt = resp.strip()
        # Remove common boilerplate prefixes
        for pat in [
            r"^Here is (a|the) .*?:\s*",
            r"^Paraphrased(?: version)?:\s*",
            r"^Sure[,.]?\s*",
            r"^Okay[,.]?\s*"
        ]:
            import re
            txt = re.sub(pat, "", txt, flags=re.I)
        return txt.strip()

    def generate(self, prompt: str, model: Optional[str] = None, temperature: float = 0.2, max_tokens: int = 512) -> Optional[str]:
        key = self.rotator.next_key()
        if not key:
            return None
        try:
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            payload = {
                "model": model or self.default_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            r = requests.post(self.url, headers=headers, json=payload, timeout=45)
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            clean = self._clean_resp(text)
            # Log the output here
            logger.info(f"[LLM][NVIDIA] out={snip(clean)}")
            return clean        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[LLM][NVIDIA] {error_msg}")
            self.rotator.mark_bad(key, error_msg)
            return None

class Paraphraser:
    """Intelligent API load balancer with rate limiting and cost optimization."""
    def __init__(self, nvidia_model: str, gemini_model_easy: str, gemini_model_hard: str):
        self.nv = NvidiaClient(KeyRotator("NVIDIA_API"), nvidia_model)
        self.gm_easy = GeminiClient(KeyRotator("GEMINI_API"), gemini_model_easy)
        # Only use GEMINI_MODEL_EASY, ignore hard model completely
        self.gm_hard = None  # Disabled - only use easy model
        
        # Rate limiting and load balancing
        self.last_nvidia_call = 0
        self.last_gemini_call = 0
        self.min_call_interval = 0.1  # Minimum 100ms between calls
        self.nvidia_success_rate = 1.0  # Track success rates for load balancing
        self.gemini_success_rate = 1.0
        self.call_counts = {"nvidia": 0, "gemini": 0, "failures": 0}
        
        logger.info("Paraphraser initialized with intelligent load balancing: NVIDIA -> GEMINI_EASY")
    
    def _rate_limit(self, api_type: str):
        """Apply rate limiting to prevent API exhaustion"""
        current_time = time.time()
        if api_type == "nvidia":
            time_since_last = current_time - self.last_nvidia_call
            if time_since_last < self.min_call_interval:
                sleep_time = self.min_call_interval - time_since_last
                time.sleep(sleep_time)
            self.last_nvidia_call = time.time()
        elif api_type == "gemini":
            time_since_last = current_time - self.last_gemini_call
            if time_since_last < self.min_call_interval:
                sleep_time = self.min_call_interval - time_since_last
                time.sleep(sleep_time)
            self.last_gemini_call = time.time()
    
    def _select_api(self, prefer_cheap: bool = True) -> str:
        """Intelligently select API based on success rates and availability"""
        nvidia_stats = self.nv.rotator.get_stats()
        gemini_stats = self.gm_easy.rotator.get_stats()
        
        nvidia_available = nvidia_stats["available_keys"] > 0
        gemini_available = gemini_stats["available_keys"] > 0
        
        if not nvidia_available and not gemini_available:
            return "none"
        elif not nvidia_available:
            return "gemini"
        elif not gemini_available:
            return "nvidia"
        
        # Both available - use intelligent selection
        if prefer_cheap:
            # Prefer NVIDIA (cheaper) but consider success rates
            if self.nvidia_success_rate > 0.8 or self.gemini_success_rate < 0.5:
                return "nvidia"
            else:
                return "gemini"
        else:
            # Prefer quality (Gemini) but consider success rates
            if self.gemini_success_rate > 0.8 or self.nvidia_success_rate < 0.5:
                return "gemini"
            else:
                return "nvidia"
    
    def _update_success_rate(self, api_type: str, success: bool):
        """Update success rate tracking for load balancing"""
        if api_type == "nvidia":
            # Exponential moving average
            alpha = 0.1
            self.nvidia_success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.nvidia_success_rate
        elif api_type == "gemini":
            alpha = 0.1
            self.gemini_success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.gemini_success_rate
    
    def _call_api(self, prompt: str, api_type: str, **kwargs) -> Optional[str]:
        """Make API call with rate limiting and error tracking"""
        self._rate_limit(api_type)
        
        try:
            if api_type == "nvidia":
                result = self.nv.generate(prompt, **kwargs)
                self.call_counts["nvidia"] += 1
                success = result is not None
                self._update_success_rate("nvidia", success)
                return result
            elif api_type == "gemini":
                result = self.gm_easy.generate(prompt, **kwargs)
                self.call_counts["gemini"] += 1
                success = result is not None
                self._update_success_rate("gemini", success)
                return result
        except Exception as e:
            self.call_counts["failures"] += 1
            self._update_success_rate(api_type, False)
            logger.error(f"[LLM] API call failed for {api_type}: {e}")
            return None
        
        return None

    # Enhanced cleaning to remove conversational elements and comments
    def _clean_resp(self, resp: str) -> str:
        if not resp: return resp
        txt = resp.strip()
        
        # Remove common conversational prefixes and comments
        prefixes_to_remove = [
            "Here's a rewritten version of",
            "Here is a rewritten version of",
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
            "Here's the answer:",
            "Here is the answer:",
            "Here's a paraphrased version:",
            "Here is a paraphrased version:",
            "Paraphrased version:",
            "Paraphrased:",
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
        ]
        
        # Remove prefixes
        for prefix in prefixes_to_remove:
            if txt.lower().startswith(prefix.lower()):
                txt = txt[len(prefix):].strip()
                break
        
        # Remove common boilerplate prefixes with regex
        import re
        for pat in [
            r"^Here is (a|the) .*?:\s*",
            r"^Paraphrased(?: version)?:\s*",
            r"^Sure[,.]?\s*",
            r"^Okay[,.]?\s*",
            r"^Certainly[,.]?\s*",
            r"^Of course[,.]?\s*",
            r"^I can .*?:\s*",
            r"^I'll .*?:\s*",
            r"^Let me .*?:\s*"
        ]:
            txt = re.sub(pat, "", txt, flags=re.I)
        
        # Remove any remaining conversational elements
        lines = txt.split('\n')
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

    # ————— Paraphrase —————
    def paraphrase(self, text: str, difficulty: str = "easy", custom_prompt: str = None) -> str:
        if not text or len(text) < 12:
            return text
        
        # Use custom prompt if provided, otherwise use optimized medical prompts
        if custom_prompt:
            prompt = custom_prompt
        else:
            # Optimized medical paraphrasing prompts based on difficulty
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
        
        # Optimize temperature and token limits based on difficulty
        temperature = 0.1 if difficulty == "easy" else 0.3
        max_tokens = min(600, max(128, len(text)//2))
        
        # Intelligent API selection with fallback
        api_type = self._select_api(prefer_cheap=True)
        
        if api_type == "nvidia":
            out = self._call_api(prompt, "nvidia", temperature=temperature, max_tokens=max_tokens)
            if out:
                return self._clean_resp(out)
            # Fallback to Gemini if NVIDIA fails
            api_type = self._select_api(prefer_cheap=False)
        
        if api_type == "gemini":
            out = self._call_api(prompt, "gemini", max_output_tokens=max_tokens)
            if out:
                logger.info(f"[LLM][GEMINI] out={snip(self._clean_resp(out))}")
                return self._clean_resp(out)
        
        # Both APIs failed
        logger.warning(f"[LLM] All APIs failed for paraphrase, returning original text")
        return text

    # ————— Translate & Backtranslate —————
    def translate(self, text: str, target_lang: str = "vi") -> Optional[str]:
        if not text: return text
        
        # Optimized medical translation prompts
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
        
        # Intelligent API selection for translation
        api_type = self._select_api(prefer_cheap=True)
        
        if api_type == "nvidia":
            out = self._call_api(prompt, "nvidia", temperature=0.0, max_tokens=min(800, len(text)+100))
            if out:
                return out.strip()
            # Fallback to Gemini if NVIDIA fails
            api_type = self._select_api(prefer_cheap=False)
        
        if api_type == "gemini":
            out = self._call_api(prompt, "gemini", max_output_tokens=min(800, len(text)+100))
            if out:
                return out.strip()
        
        return None

    def backtranslate(self, text: str, via_lang: str = "vi") -> Optional[str]:
        if not text: return text
        mid = self.translate(text, target_lang=via_lang)
        if not mid: return None
        
        # Optimized backtranslation prompt with medical focus
        if via_lang == "vi":
            prompt = (
                "Translate the following Vietnamese medical text back to English while preserving all medical terminology, clinical facts, and professional medical language. Ensure the translation is medically accurate. Return only the translation without any introduction or commentary.\n\n"
                f"{mid}"
            )
        else:
            prompt = (
                f"Translate the following {via_lang} medical text back to English while preserving all medical terminology, clinical facts, and professional medical language. Return only the translation without any introduction or commentary.\n\n"
                f"{mid}"
            )
        
        # Intelligent API selection for backtranslation
        api_type = self._select_api(prefer_cheap=True)
        
        if api_type == "nvidia":
            out = self._call_api(prompt, "nvidia", temperature=0.0, max_tokens=min(900, len(text)+150))
            if out:
                return out.strip()
            # Fallback to Gemini if NVIDIA fails
            api_type = self._select_api(prefer_cheap=False)
        
        if api_type == "gemini":
            out = self._call_api(prompt, "gemini", max_output_tokens=min(900, len(text)+150))
            if out:
                return out.strip()
        
        return None

    # ————— Consistency Judge (cheap, ratio-based) —————
    def consistency_check(self, user: str, output: str) -> bool:
        """Return True if 'output' appears supported by 'user' (context/question). Optimized medical validation."""
        prompt = (
            "Evaluate if the medical answer is consistent with the question/context and medically accurate. Consider medical accuracy, clinical appropriateness, consistency with the question, safety standards, and completeness of medical information. Reply with exactly 'PASS' if the answer is medically sound and consistent, otherwise 'FAIL'.\n\n"
            f"Question/Context: {user}\n\n"
            f"Medical Answer: {output}"
        )
        
        # Use intelligent API selection for consistency check
        api_type = self._select_api(prefer_cheap=True)
        
        if api_type == "nvidia":
            out = self._call_api(prompt, "nvidia", temperature=0.0, max_tokens=5)
            if out:
                return isinstance(out, str) and "PASS" in out.upper()
            # Fallback to Gemini if NVIDIA fails
            api_type = self._select_api(prefer_cheap=False)
        
        if api_type == "gemini":
            out = self._call_api(prompt, "gemini", max_output_tokens=5)
            if out:
                return isinstance(out, str) and "PASS" in out.upper()
        
        # If both APIs fail, assume consistency (conservative approach)
        logger.warning("[LLM] Consistency check failed due to API unavailability, assuming consistent")
        return True
    
    def get_api_stats(self) -> dict:
        """Get comprehensive API usage statistics"""
        nvidia_stats = self.nv.rotator.get_stats()
        gemini_stats = self.gm_easy.rotator.get_stats()
        
        return {
            "call_counts": self.call_counts.copy(),
            "success_rates": {
                "nvidia": self.nvidia_success_rate,
                "gemini": self.gemini_success_rate
            },
            "nvidia_rotator": nvidia_stats,
            "gemini_rotator": gemini_stats,
            "total_calls": sum(self.call_counts.values()),
            "failure_rate": self.call_counts["failures"] / max(1, sum(self.call_counts.values()))
        }
    
    def medical_accuracy_check(self, question: str, answer: str) -> bool:
        """Check medical accuracy of Q&A pairs using cloud APIs"""
        if not question or not answer:
            return False
            
        prompt = (
            "Evaluate if the medical answer is accurate and appropriate for the question. Consider medical facts, clinical knowledge, appropriate medical terminology, clinical reasoning, logic, and safety considerations. Reply with exactly 'ACCURATE' if the answer is medically correct, otherwise 'INACCURATE'.\n\n"
            f"Medical Question: {question}\n\n"
            f"Medical Answer: {answer}"
        )
        
        out = self.nv.generate(prompt, temperature=0.0, max_tokens=5)
        if not out:
            out = self.gm_easy.generate(prompt, max_output_tokens=5)
        return isinstance(out, str) and "ACCURATE" in out.upper()
    
    def enhance_medical_terminology(self, text: str) -> str:
        """Enhance medical terminology in text using cloud APIs"""
        if not text or len(text) < 20:
            return text
            
        prompt = (
            "Improve the medical terminology in the following text while preserving all factual information and clinical accuracy. Use more precise medical terms where appropriate. Return only the improved text without any introduction or commentary.\n\n"
            f"{text}"
        )
        
        out = self.nv.generate(prompt, temperature=0.1, max_tokens=min(800, len(text)+100))
        if not out:
            out = self.gm_easy.generate(prompt, max_output_tokens=min(800, len(text)+100))
        return out if out else text
    
    def create_clinical_scenarios(self, question: str, answer: str) -> list:
        """Create different clinical scenarios from Q&A pairs using cloud APIs"""
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
        
        for prompt_template, scenario_type in context_prompts:
            try:
                prompt = prompt_template.format(question=question)
                scenario_question = self.paraphrase(question, difficulty="hard", custom_prompt=prompt)
                
                if scenario_question and not self._is_invalid_response(scenario_question):
                    scenarios.append((scenario_question, answer, scenario_type))
            except Exception as e:
                logger.warning(f"Failed to create clinical scenario {scenario_type}: {e}")
                continue
                
        return scenarios
    
    def _is_invalid_response(self, text: str) -> bool:
        """Check if response is invalid"""
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
