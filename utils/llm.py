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
            logger.info(f"[LLM][NVIDIA] out={snip(clean)}")
            return clean        
        except Exception as e:
            logger.error(f"[LLM][NVIDIA] {e}")
            self.rotator.mark_bad(key)
            return None

class Paraphraser:
    """Prefers NVIDIA (cheap), falls back to Gemini. Also offers translate/backtranslate and a tiny consistency judge."""
    def __init__(self, nvidia_model: str, gemini_model_easy: str, gemini_model_hard: str):
        self.nv = NvidiaClient(KeyRotator("NVIDIA_API"), nvidia_model)
        self.gm_easy = GeminiClient(KeyRotator("GEMINI_API"), gemini_model_easy)
        self.gm_hard = GeminiClient(KeyRotator("GEMINI_API"), gemini_model_hard)

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
    def paraphrase(self, text: str, difficulty: str = "easy") -> str:
        if not text or len(text) < 12:
            return text
        prompt = (
            "Paraphrase the following medical text concisely, preserve meaning and clinical terms.\n"
            "Do not fabricate or remove factual claims.\n" 
            "Return ONLY the rewritten text, without any introduction, commentary.\n"+ text
        )
        out = self.nv.generate(prompt, temperature=0.1, max_tokens=min(600, max(128, len(text)//2)))
        if out: return self._clean_resp(out)
        gm = self.gm_easy if difficulty == "easy" else self.gm_hard
        out = gm.generate(prompt, max_output_tokens=min(600, max(128, len(text)//2)))
        return self._clean_resp(out) if out else text

    # ————— Translate & Backtranslate —————
    def translate(self, text: str, target_lang: str = "de") -> Optional[str]:
        if not text: return text
        prompt = f"Translate to {target_lang}. Keep meaning exact, preserve medical terms:\n\n{text}"
        out = self.nv.generate(prompt, temperature=0.0, max_tokens=min(800, len(text)+100))
        if out: return out.strip()
        return self.gm_easy.generate(prompt, max_output_tokens=min(800, len(text)+100))

    def backtranslate(self, text: str, via_lang: str = "de") -> Optional[str]:
        if not text: return text
        mid = self.translate(text, target_lang=via_lang)
        if not mid: return None
        prompt = f"Translate the following {via_lang} text back to English, preserving the exact meaning:\n\n{mid}"
        out = self.nv.generate(prompt, temperature=0.0, max_tokens=min(900, len(text)+150))
        if out: return out.strip()
        res = self.gm_easy.generate(prompt, max_output_tokens=min(900, len(text)+150))
        return res.strip() if res else None

    # ————— Consistency Judge (cheap, ratio-based) —————
    def consistency_check(self, user: str, output: str) -> bool:
        """Return True if 'output' appears supported by 'user' (context/question). Soft heuristic via LLM."""
        prompt = (
            "You are a strict medical QA validator. Given the USER input (question+context) "
            "and the MODEL ANSWER, reply with exactly 'PASS' if the answer is supported and safe, "
            "otherwise 'FAIL'. No extra text.\n\n"
            f"USER:\n{user}\n\nANSWER:\n{output}"
        )
        out = self.nv.generate(prompt, temperature=0.0, max_tokens=3)
        if not out:
            out = self.gm_easy.generate(prompt, max_output_tokens=3)
        return isinstance(out, str) and "PASS" in out.upper()
