# augmentation utility agent
import re
import difflib
import random
from typing import Dict, Tuple
import ftfy
import langid
import logging

# Module logger
logger = logging.getLogger("augment")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

P_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
P_PHONE = re.compile(r"(?:(?:\+?\d{1,3})?[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}")
P_URL   = re.compile(r"https?://\S+|www\.\S+")
P_IP    = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

def fix_unicode(s: str) -> str:
    return ftfy.fix_text(s or "")

def normalize_whitespace(s: str) -> str:
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def canonicalize_quotes(s: str) -> str:
    return s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

def ensure_terminal_punct(s: str) -> str:
    if not s: return s
    if s[-1] in ".!?": return s
    return s + "."

def deidentify(s: str) -> str:
    s = P_EMAIL.sub("[REDACTED_EMAIL]", s)
    s = P_PHONE.sub("[REDACTED_PHONE]", s)
    s = P_URL.sub("[REDACTED_URL]", s)
    s = P_IP.sub("[REDACTED_IP]", s)
    return s

def lang_is_english(s: str) -> bool:
    try:
        lang, _ = langid.classify((s or "")[:2000])
        return lang == "en"
    except Exception:
        return True

def length_cap(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    # try to cut at sentence boundary
    cut = s[:max_chars]
    last_dot = cut.rfind(". ")
    if last_dot > 300:  # don't cut too aggressively
        return cut[:last_dot+1] + " …"
    return cut + " …"

def fingerprint(instr: str, user: str, out: str) -> str:
    # Simple, fast fingerprint for dedupe
    def norm(x: str) -> str:
        x = x.lower()
        x = re.sub(r"[^a-z0-9]+", " ", x)
        x = re.sub(r"\s+", " ", x).strip()
        return x
    core = "||".join([norm(instr), norm(user), norm(out)])
    # lightweight hash
    import hashlib
    return hashlib.md5(core.encode("utf-8")).hexdigest()

def style_standardize_answer(ans: str) -> str:
    if not ans: return ans
    ans = ans.strip()
    # Gentle guardrails, neutral voice
    prefix = ""
    # Avoid absolute guarantees
    ans = re.sub(r"\b(guarantee|100%|certainly|always|never)\b", "likely", ans, flags=re.I)
    # Remove sign-offs typical of forums
    ans = re.sub(r"\n*(thanks|thank you|regards|cheers)[^\n]*$", "", ans, flags=re.I)
    return ensure_terminal_punct(ans)

def base_cleanup(s: str, max_chars: int, do_deid: bool) -> str:
    s = fix_unicode(s)
    s = canonicalize_quotes(s)
    s = normalize_whitespace(s)
    if do_deid:
        s = deidentify(s)
    s = length_cap(s, max_chars)
    return s

def maybe_paraphrase(text: str, ratio: float, paraphraser, difficulty: str) -> Tuple[str, bool]:
    if ratio <= 0 or not text: return text, False
    if random.random() < ratio:
        return paraphraser.paraphrase(text, difficulty=difficulty), True
    return text, False

def maybe_backtranslate(text: str, ratio: float, paraphraser) -> Tuple[str, bool]:
    if ratio <= 0 or not text: return text, False
    if random.random() < ratio:
        bt = paraphraser.backtranslate(text, via_lang="vi")
        if not bt:
            return text, False
        # Guardrails: reject if too short/long or too dissimilar/similar
        try:
            orig_len = max(1, len(text))
            len_delta = abs(len(bt) - len(text)) / orig_len
            sim = difflib.SequenceMatcher(None, text, bt).ratio()
            # Accept if moderate change and not excessive drift
            if len_delta > 0.5:
                return text, False
            if sim < 0.45 or sim > 0.98:
                return text, False
        except Exception:
            pass
        return bt, True
    return text, False

def consistency_ok(user: str, out: str, ratio: float, paraphraser) -> bool:
    if ratio <= 0 or (not user) or (not out):
        return True
    if random.random() >= ratio:
        return True
    return paraphraser.consistency_check(user, out)

def is_invalid_response(text: str) -> bool:
    """Check if model response is invalid (Fail, Invalid, etc.)"""
    if not text or not isinstance(text, str):
        return True
    
    text_lower = text.lower().strip()
    invalid_patterns = [
        "fail", "invalid", "i couldn't", "i can't", "i cannot", "unable to",
        "sorry", "error", "not available", "no answer", "insufficient",
        "don't know", "do not know", "not sure", "cannot determine",
        "unable to provide", "not possible", "not applicable", "n/a"
    ]
    
    # Check if response is too short or matches invalid patterns
    if len(text_lower) < 3:
        return True
    
    for pattern in invalid_patterns:
        if pattern in text_lower:
            return True
    
    return False

def clean_conversational_elements(text: str) -> str:
    """Remove conversational elements and non-medical information smartly"""
    if not text or not isinstance(text, str):
        return text
    
    # Remove common conversational prefixes
    conversational_prefixes = [
        r"^(hi|hello|hey|greetings?)\s*,?\s*",
        r"^(xin chào|chào|chào bạn)\s*,?\s*",
        r"^(if you are a doctor|if you're a doctor|as a doctor)\s*,?\s*",
        r"^(nếu bạn là bác sĩ|nếu bạn là doctor)\s*,?\s*",
        r"^(please|vui lòng)\s*,?\s*",
        r"^(thank you|cảm ơn)\s*,?\s*",
        r"^(thanks|cảm ơn)\s*,?\s*",
        r"^(regards|best regards|cheers)\s*,?\s*",
        r"^(i hope this helps|hy vọng điều này giúp ích)\s*,?\s*",
        r"^(i'm sorry|tôi xin lỗi)\s*,?\s*",
        r"^(let me help|để tôi giúp)\s*,?\s*",
        r"^(i understand|tôi hiểu)\s*,?\s*",
        r"^(i can help|tôi có thể giúp)\s*,?\s*",
        r"^(i'll be happy to|tôi sẽ vui lòng)\s*,?\s*",
        r"^(i would be glad to|tôi sẽ rất vui)\s*,?\s*",
        r"^(i'm here to help|tôi ở đây để giúp)\s*,?\s*",
        r"^(i'm a doctor|tôi là bác sĩ)\s*,?\s*",
        r"^(as a medical professional|như một chuyên gia y tế)\s*,?\s*",
        r"^(from a medical perspective|từ góc độ y tế)\s*,?\s*",
        r"^(medically speaking|nói về mặt y tế)\s*,?\s*",
    ]
    
    cleaned_text = text
    for pattern in conversational_prefixes:
        import re
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)
    
    # Remove common conversational suffixes
    conversational_suffixes = [
        r"\s*,?\s*(hope this helps|hy vọng điều này giúp ích).*$",
        r"\s*,?\s*(let me know if you need more|hãy cho tôi biết nếu bạn cần thêm).*$",
        r"\s*,?\s*(feel free to ask|đừng ngại hỏi).*$",
        r"\s*,?\s*(if you have any questions|nếu bạn có câu hỏi).*$",
        r"\s*,?\s*(please let me know|vui lòng cho tôi biết).*$",
        r"\s*,?\s*(i'm here to help|tôi ở đây để giúp).*$",
        r"\s*,?\s*(best regards|trân trọng).*$",
        r"\s*,?\s*(take care|chúc sức khỏe).*$",
        r"\s*,?\s*(good luck|chúc may mắn).*$",
        r"\s*,?\s*(wishing you well|chúc bạn khỏe mạnh).*$",
    ]
    
    for pattern in conversational_suffixes:
        import re
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)
    
    # Clean up extra whitespace and punctuation
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    cleaned_text = re.sub(r'^[,\s]+|[,\s]+$', '', cleaned_text)
    
    return cleaned_text if cleaned_text else text

def clean_invalid_response(text: str, fallback: str = "") -> str:
    """Clean invalid responses by returning fallback or empty string"""
    if is_invalid_response(text):
        return fallback
    return text

def retry_invalid_response(text: str, paraphraser, max_retries: int = 3) -> str:
    """Retry generating valid response for invalid text, max 3 retries"""
    if not is_invalid_response(text):
        return text
    
    # Clean conversational elements first
    cleaned_text = clean_conversational_elements(text)
    if cleaned_text != text and not is_invalid_response(cleaned_text):
        return cleaned_text
    
    for attempt in range(max_retries):
        try:
            # Try different strategies based on attempt
            if attempt == 0:
                # First try: Simple paraphrasing
                retry_text = paraphraser.paraphrase(text, difficulty="easy")
            elif attempt == 1:
                # Second try: More aggressive paraphrasing with medical focus
                medical_prompt = f"Rewrite this medical response to be more professional and accurate. Return only the rewritten response without any introduction or commentary:\n\n{text}"
                retry_text = paraphraser.paraphrase(text, difficulty="hard", custom_prompt=medical_prompt)
            else:
                # Third try: Direct medical content generation
                medical_prompt = f"Provide a professional medical response to this question. Return only the medical response without any introduction or commentary:\n\n{text}"
                retry_text = paraphraser.paraphrase(text, difficulty="hard", custom_prompt=medical_prompt)
            
            if retry_text and not is_invalid_response(retry_text):
                # Clean conversational elements from retry
                cleaned_retry = clean_conversational_elements(retry_text)
                if cleaned_retry and not is_invalid_response(cleaned_retry):
                    return cleaned_retry
                elif retry_text:  # Use original retry if cleaning fails
                    return retry_text
                    
        except Exception as e:
            logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
            continue
    
    # If all retries failed, return empty string to indicate drop
    return ""

def validate_medical_accuracy(question: str, answer: str, paraphraser) -> bool:
    """Validate medical accuracy of Q&A pairs using LLM consistency check"""
    if not question or not answer:
        return False
    
    try:
        # Use medical accuracy check if available (local mode), otherwise fallback to consistency check
        if hasattr(paraphraser, 'medical_accuracy_check'):
            return paraphraser.medical_accuracy_check(question, answer)
        else:
            return paraphraser.consistency_check(question, answer)
    except Exception as e:
        logger.warning(f"Medical accuracy validation failed: {e}")
        return True  # Default to accepting if validation fails

def enhance_medical_terminology(text: str, paraphraser) -> str:
    """Enhance medical terminology in text while preserving accuracy"""
    if not text or len(text) < 20:
        return text
    
    try:
        # Use dedicated method if available (local mode), otherwise use paraphrase with custom prompt
        if hasattr(paraphraser, 'enhance_medical_terminology'):
            enhanced = paraphraser.enhance_medical_terminology(text)
            if enhanced and not is_invalid_response(enhanced):
                return enhanced
        else:
            prompt = (
                "Improve the medical terminology in this text while preserving all factual information. Return only the improved text with better medical terminology without any introduction or commentary:\n\n"
                f"{text}"
            )
            
            enhanced = paraphraser.paraphrase(text, difficulty="hard", custom_prompt=prompt)
            if enhanced and not is_invalid_response(enhanced):
                return enhanced
    except Exception as e:
        logger.warning(f"Medical terminology enhancement failed: {e}")
    
    return text

def create_clinical_scenarios(question: str, answer: str, paraphraser) -> list:
    """Create different clinical scenarios from a Q&A pair"""
    scenarios = []
    
    try:
        # Use dedicated method if available (local mode), otherwise use paraphrase with custom prompts
        if hasattr(paraphraser, 'create_clinical_scenarios'):
            scenarios = paraphraser.create_clinical_scenarios(question, answer)
        else:
            # Fallback to original implementation
            context_prompts = [
                f"Rewrite this medical question as if asked by a patient in an emergency room. Return only the rewritten question without any introduction or commentary:\n\n{question}",
                f"Rewrite this medical question as if asked by a patient in a routine checkup. Return only the rewritten question without any introduction or commentary:\n\n{question}",
                f"Rewrite this medical question as if asked by a patient with chronic conditions. Return only the rewritten question without any introduction or commentary:\n\n{question}",
                f"Rewrite this medical question as if asked by a patient's family member. Return only the rewritten question without any introduction or commentary:\n\n{question}"
            ]
            
            for i, prompt in enumerate(context_prompts):
                try:
                    scenario_question = paraphraser.paraphrase(question, difficulty="hard", custom_prompt=prompt)
                    if scenario_question and not is_invalid_response(scenario_question):
                        scenarios.append((scenario_question, answer, f"clinical_scenario_{i+1}"))
                except Exception as e:
                    logger.warning(f"Failed to create clinical scenario {i+1}: {e}")
                    continue
                
    except Exception as e:
        logger.warning(f"Clinical scenario creation failed: {e}")
    
    return scenarios
