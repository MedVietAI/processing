# augmentation utility agent
import re
import random
from typing import Dict, Tuple
import ftfy
import langid

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
        bt = paraphraser.backtranslate(text, via_lang="de")
        return bt if bt else text, bool(bt)
    return text, False

def consistency_ok(user: str, out: str, ratio: float, paraphraser) -> bool:
    if ratio <= 0 or (not user) or (not out):
        return True
    if random.random() >= ratio:
        return True
    return paraphraser.consistency_check(user, out)
