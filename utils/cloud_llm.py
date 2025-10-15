# Round-robin rotator + paraphrasing + translation/backtranslation
import os
import logging
import requests
from typing import Optional
from google import genai

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
        self.dead = set()
        self.idx = 0

    def next_key(self) -> Optional[str]:
        if not self.keys:
            return None
        for _ in range(len(self.keys)):
            k = self.keys[self.idx % len(self.keys)]
            self.idx += 1
            if k not in self.dead:
                return k
        return None

    def mark_bad(self, key: Optional[str]):
        if key:
            self.dead.add(key)
            logger.warning(f"[LLM] Quarantined key (prefix hidden): {key[:6]}***")

class GeminiClient:
    def __init__(self, rotator: KeyRotator, default_model: str):
        self.rotator = rotator
        self.default_model = default_model

    def generate(self, prompt: str, model: Optional[str] = None, temperature: float = 0.2, max_output_tokens: int = 512) -> Optional[str]:
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
            logger.error(f"[LLM][Gemini] {e}")
            self.rotator.mark_bad(key)
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
            logger.error(f"[LLM][NVIDIA] {e}")
            self.rotator.mark_bad(key)
            return None

class Paraphraser:
    """Prefers NVIDIA (cheap), falls back to Gemini EASY only. Also offers translate/backtranslate and a tiny consistency judge."""
    def __init__(self, nvidia_model: str, gemini_model_easy: str, gemini_model_hard: str):
        self.nv = NvidiaClient(KeyRotator("NVIDIA_API"), nvidia_model)
        self.gm_easy = GeminiClient(KeyRotator("GEMINI_API"), gemini_model_easy)
        # Only use GEMINI_MODEL_EASY, ignore hard model completely
        self.gm_hard = None  # Disabled - only use easy model
        logger.info("Paraphraser initialized: NVIDIA -> GEMINI_EASY (GEMINI_HARD disabled)")

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
        
        # Always try NVIDIA first (optimized for medical tasks)
        out = self.nv.generate(prompt, temperature=temperature, max_tokens=max_tokens)
        if out: 
            return self._clean_resp(out)
        
        # Fallback to GEMINI with optimized parameters
        out = self.gm_easy.generate(prompt, max_output_tokens=max_tokens)
        if out:
            logger.info(f"[LLM][GEMINI] out={snip(self._clean_resp(out))}")
            return self._clean_resp(out)
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
        
        out = self.nv.generate(prompt, temperature=0.0, max_tokens=min(800, len(text)+100))
        if out: return out.strip()
        return self.gm_easy.generate(prompt, max_output_tokens=min(800, len(text)+100))

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
        
        out = self.nv.generate(prompt, temperature=0.0, max_tokens=min(900, len(text)+150))
        if out: return out.strip()
        res = self.gm_easy.generate(prompt, max_output_tokens=min(900, len(text)+150))
        return res.strip() if res else None

    # ————— Consistency Judge (cheap, ratio-based) —————
    def consistency_check(self, user: str, output: str) -> bool:
        """Return True if 'output' appears supported by 'user' (context/question). Optimized medical validation."""
        prompt = (
            "Evaluate if the medical answer is consistent with the question/context and medically accurate. Consider medical accuracy, clinical appropriateness, consistency with the question, safety standards, and completeness of medical information. Reply with exactly 'PASS' if the answer is medically sound and consistent, otherwise 'FAIL'.\n\n"
            f"Question/Context: {user}\n\n"
            f"Medical Answer: {output}"
        )
        out = self.nv.generate(prompt, temperature=0.0, max_tokens=5)
        if not out:
            out = self.gm_easy.generate(prompt, max_output_tokens=5)
        return isinstance(out, str) and "PASS" in out.upper()
    
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
