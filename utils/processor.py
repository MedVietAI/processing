# Dataset-specific parsers + paraphrasing flow
import json
import random
import hashlib
import logging
from typing import Callable, Optional, Dict, Tuple

from utils.schema import sft_row
from utils import augment as A
from vi.processing import translate_sft_row, should_translate, log_translation_stats

# Logger
logger = logging.getLogger("processor")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())


def _hash_id(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
    return h.hexdigest()[:16]

def _iter_json_or_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1); f.seek(0)
        if first == "[":
            data = json.load(f)
            for obj in data: yield obj
        else:
            for line in f:
                line = line.strip()
                if line: yield json.loads(line)

def process_file_into_sft(
    dataset_key: str,
    input_path: str,
    writer,
    paraphraser,
    augment_opts: Dict,
    sample_limit: Optional[int],
    seed: int,
    progress_cb: Optional[Callable[[float, str], None]],
    translator=None
) -> Tuple[int, Dict]:
    random.seed(seed)
    stats = {
        "written": 0,
        "paraphrased_input": 0,
        "paraphrased_output": 0,
        "backtranslated_input": 0,
        "backtranslated_output": 0,
        "dedup_skipped": 0,
        "consistency_failed": 0,
        "medical_accuracy_failed": 0,
        "clinical_scenarios_created": 0,
        "enhanced_terminology": 0,
        "vietnamese_variants": 0
    }
    # Start processing SFT
    key_summary = {k: augment_opts.get(k) for k in (
        "paraphrase_ratio","backtranslate_ratio","paraphrase_outputs",
        "style_standardize","deidentify","dedupe",
        "consistency_check_ratio","distill_fraction"
    )}
    logger.info(
        f"[PROC] Begin dataset={dataset_key} sample_limit={sample_limit} opts={key_summary}"
    )
    # If deduplicating enabled
    dedupe_seen = set() if augment_opts.get("dedupe", True) else None

    key = dataset_key.lower()
    if key in ("healthcaremagic", "icliniq"):
        count = _proc_med_dialog(source=key, path=input_path, writer=writer,
                                 paraphraser=paraphraser, opts=augment_opts,
                                 sample_limit=sample_limit, stats=stats, cb=progress_cb, dedupe_seen=dedupe_seen, translator=translator)
    elif key == "pubmedqa_l":
        count = _proc_pubmedqa_l(input_path, writer, paraphraser, augment_opts, sample_limit, stats, progress_cb, dedupe_seen=dedupe_seen, translator=translator)
    elif key == "pubmedqa_u":
        count = _proc_pubmedqa_u(input_path, writer, paraphraser, augment_opts, sample_limit, stats, progress_cb, dedupe_seen=dedupe_seen, translator=translator)
    elif key == "pubmedqa_map":
        count = _proc_pubmedqa_map(input_path, writer, paraphraser, augment_opts, sample_limit, stats, progress_cb, dedupe_seen=dedupe_seen, translator=translator)
    else:
        raise ValueError(f"Unknown dataset: {dataset_key}")
    logger.info(f"[PROC] End dataset={dataset_key} stats={stats}")
    return count, stats

# ——————————— helpers ———————————
def _build_variants(user: str, out: str, paraphraser, opts: Dict, stats: Dict):
    """Return a list of (user_variant, out_variant, applied_tags) not including the original."""
    variants = []
    max_k = max(0, int(opts.get("max_aug_per_sample", 1)))
    for _ in range(max_k):
        applied = []
        u2, did_p = A.maybe_paraphrase(user, opts.get("paraphrase_ratio", 0.0), paraphraser, "easy")
        if did_p: applied.append("paraphrase_input"); stats["paraphrased_input"] += 1
        u3, did_bt = A.maybe_backtranslate(u2, opts.get("backtranslate_ratio", 0.0), paraphraser)
        if did_bt: applied.append("backtranslate_input"); stats["backtranslated_input"] += 1

        o3 = out
        if opts.get("paraphrase_outputs", False):
            o2, did_p2 = A.maybe_paraphrase(out, opts.get("paraphrase_ratio", 0.0), paraphraser, "hard")
            if did_p2: applied.append("paraphrase_output"); stats["paraphrased_output"] += 1
            o3b, did_bt2 = A.maybe_backtranslate(o2, opts.get("backtranslate_ratio", 0.0), paraphraser)
            if did_bt2: applied.append("backtranslate_output"); stats["backtranslated_output"] += 1
            o3 = o3b

        # If nothing applied, skip this variant
        if not applied:
            continue
        # Style standardize and punctuation for the variant too
        if opts.get("style_standardize", True):
            o3 = A.style_standardize_answer(o3)
        u3 = A.ensure_terminal_punct(u3) if u3 else u3
        o3 = A.ensure_terminal_punct(o3) if o3 else o3
        variants.append((u3, o3, applied))
    return variants

def _build_enriched_variants(user: str, out: str, paraphraser, opts: Dict, stats: Dict, translator=None):
    """Build multiple paraphrased variants for SFT enrichment with enhanced diversity strategies"""
    variants = []
    
    # Enhanced answer generation with different perspectives
    answer_variants = []
    answer_strategies = [
        ("original", out, ["original_answer"]),
        ("concise", None, ["concise_answer"]),
        ("detailed", None, ["detailed_answer"]),
        ("clinical", None, ["clinical_answer"]),
        ("patient_friendly", None, ["patient_friendly_answer"])
    ]
    
    for strategy, original_text, tags in answer_strategies:
        if strategy == "original":
            answer_variants.append((original_text, tags))
        else:
            try:
                # Generate different answer styles
                style_prompt = _get_answer_style_prompt(strategy, user, out)
                enhanced_out = paraphraser.paraphrase(out, difficulty="hard", custom_prompt=style_prompt)
                
                if enhanced_out and not A.is_invalid_response(enhanced_out):
                    if opts.get("style_standardize", True):
                        enhanced_out = A.style_standardize_answer(enhanced_out)
                    enhanced_out = A.ensure_terminal_punct(enhanced_out)
                    answer_variants.append((enhanced_out, tags))
                    stats["paraphrased_output"] += 1
            except Exception as e:
                logger.warning(f"Failed to generate {strategy} answer variant: {e}")
                continue
    
    # Enhanced question generation with different question types
    question_variants = []
    question_strategies = [
        ("original", user, ["original_question"]),
        ("clarifying", None, ["clarifying_question"]),
        ("follow_up", None, ["follow_up_question"]),
        ("symptom_focused", None, ["symptom_focused_question"]),
        ("treatment_focused", None, ["treatment_focused_question"])
    ]
    
    for strategy, original_text, tags in question_strategies:
        if strategy == "original":
            question_variants.append((original_text, tags))
        else:
            try:
                # Generate different question styles
                style_prompt = _get_question_style_prompt(strategy, user, out)
                enhanced_user = paraphraser.paraphrase(user, difficulty="hard", custom_prompt=style_prompt)
                
                if enhanced_user and not A.is_invalid_response(enhanced_user):
                    enhanced_user = A.ensure_terminal_punct(enhanced_user)
                    question_variants.append((enhanced_user, tags))
                    stats["paraphrased_input"] += 1
            except Exception as e:
                logger.warning(f"Failed to generate {strategy} question variant: {e}")
                continue
    
    # Create combinations: each question variant with each answer variant
    for q_user, q_tags in question_variants:
        for a_out, a_tags in answer_variants:
            combined_tags = q_tags + a_tags
            variants.append((q_user, a_out, combined_tags))
    
    # Add Vietnamese variants if translator is available
    if translator and translator.is_loaded():
        vi_variants = []
        for q_user, a_out, tags in variants[:5]:  # Limit to first 5 to avoid too many variants
            try:
                # Translate question and answer
                vi_q = translator.translate_text(q_user)
                vi_a = translator.translate_text(a_out)
                
                if vi_q and vi_a and not A.is_invalid_response(vi_q) and not A.is_invalid_response(vi_a):
                    vi_tags = tags + ["vietnamese_translated"]
                    vi_variants.append((vi_q, vi_a, vi_tags))
                    stats["vietnamese_variants"] = stats.get("vietnamese_variants", 0) + 1
            except Exception as e:
                logger.warning(f"Failed to create Vietnamese variant: {e}")
                continue
        
        variants.extend(vi_variants)
    
    return variants

def _get_answer_style_prompt(strategy: str, question: str, original_answer: str) -> str:
    """Generate style-specific prompts for answer enhancement"""
    prompts = {
        "concise": f"Rewrite this medical answer to be more concise while preserving all key medical information:\n\n{original_answer}",
        "detailed": f"Expand this medical answer with more detailed explanations while maintaining accuracy:\n\n{original_answer}",
        "clinical": f"Rewrite this answer using more formal clinical language and medical terminology:\n\n{original_answer}",
        "patient_friendly": f"Rewrite this medical answer in simpler, more patient-friendly language while keeping it medically accurate:\n\n{original_answer}"
    }
    return prompts.get(strategy, f"Paraphrase this medical answer: {original_answer}")

def _get_question_style_prompt(strategy: str, original_question: str, answer: str) -> str:
    """Generate style-specific prompts for question enhancement"""
    prompts = {
        "clarifying": f"Rewrite this medical question to ask for clarification or more specific information:\n\n{original_question}",
        "follow_up": f"Create a follow-up question that a patient might ask after this medical question:\n\n{original_question}",
        "symptom_focused": f"Rewrite this question to focus more on symptoms and their characteristics:\n\n{original_question}",
        "treatment_focused": f"Rewrite this question to focus more on treatment options and management:\n\n{original_question}"
    }
    return prompts.get(strategy, f"Paraphrase this medical question: {original_question}")

def _apply_aug(instr: str, user: str, out: str, source: str, opts: Dict, paraphraser, stats: Dict):
    # Base cleanup & caps (returns cleaned strings)
    user = A.base_cleanup(user, opts.get("max_chars", 5000), opts.get("deidentify", True))
    out  = A.base_cleanup(out,  opts.get("max_chars", 5000), opts.get("deidentify", True))
    instr = A.base_cleanup(instr, opts.get("max_chars", 5000), False)

    # Language sanity (mostly English—skip aggressive transforms if not)
    if not A.lang_is_english(user):  # very rare
        return instr, user, out, []

    # Stack list of entries that has been applied augmentation and stylings
    applied = []

    # Clean invalid responses with retry logic
    if A.is_invalid_response(out):
        out = A.retry_invalid_response(out, paraphraser, max_retries=3)
        if not out:  # If retry failed, return empty to indicate drop
            return instr, user, "", applied
        applied.append("invalid_response_retried")

    # Style standardizing the answer
    if opts.get("style_standardize", True):
        out = A.style_standardize_answer(out)
        applied.append("style_standardize")

    # Ensure punctuation/whitespace
    user = A.ensure_terminal_punct(user) if user else user
    out  = A.ensure_terminal_punct(out)  if out  else out

    return instr, user, out, applied

def _commit_row(writer, source, rid, task, instr, user, out, opts, stats, aug_applied, extra_meta=None, dedupe_seen=None, translator=None):
    # Dedup entry
    if dedupe_seen is not None:
        fp = A.fingerprint(instr, user, out)
        if fp in dedupe_seen:
            stats["dedup_skipped"] += 1
            return False
        dedupe_seen.add(fp)

    meta = {"augmentations": aug_applied}
    if extra_meta:
        meta.update(extra_meta)

    row = sft_row(instr, user, out, source=source, rid=rid, task=task, meta=meta)
    
    # Apply Vietnamese translation if requested
    if should_translate(opts.get("vietnamese_translation", False), translator):
        try:
            row = translate_sft_row(row, translator)
            meta["vietnamese_translated"] = True
            row["meta"] = meta
        except Exception as e:
            logger.error(f"Failed to translate SFT row: {e}")
    
    writer.write(row)
    stats["written"] += 1
    return True

# ——————————— dataset processors ———————————

def _proc_med_dialog(source, path, writer, paraphraser, opts, sample_limit, stats, cb, dedupe_seen=None, translator=None):
    count = 0
    written = 0
    for i, obj in enumerate(_iter_json_or_jsonl(path), start=1):
        try:
            instr_raw = obj.get("instruction") or "Answer the patient's question like a clinician. Be concise and safe."
            user_raw = obj.get("input") or ""
            out_raw = obj.get("output") or ""
            
            # Ensure we have string values
            instr = str(instr_raw).strip()
            user = str(user_raw).strip()
            out = str(out_raw).strip()
            rid = _hash_id(source, i, len(user), len(out))
        except Exception as e:
            logger.warning(f"[PROC] {source} error processing item {i}: {e}, item: {obj}")
            continue

        try:
            instr, user, out, applied = _apply_aug(instr, user, out, source, opts, paraphraser, stats)

            # Skip if retry failed (empty output)
            if not out:
                stats["dropped_invalid"] = stats.get("dropped_invalid", 0) + 1
                continue

            # 1) ALWAYS write the original (cleaned/style-standardised only)
            # Enhanced medical accuracy validation
            if not A.validate_medical_accuracy(user, out, paraphraser):
                stats["medical_accuracy_failed"] = stats.get("medical_accuracy_failed", 0) + 1
                applied.append("medical_accuracy_flag")
            
            # Optional consistency spot-check (cheap)
            if not A.consistency_ok(user, out, opts.get("consistency_check_ratio", 0.0), paraphraser):
                stats["consistency_failed"] += 1
                # keep the sample but tag it
                applied.append("consistency_flag")

            # 2) If expansion is enabled, add enriched variants for SFT
            _commit_row(writer, source, rid, "medical_dialogue", instr, user, out, opts, stats, ["base"] + applied, dedupe_seen=dedupe_seen, translator=translator)
            
            # Add enriched variants if expand is enabled
            if opts.get("expand", True):
                # Use enriched variants for SFT (multiple Q&A combinations)
                enriched_variants = _build_enriched_variants(user, out, paraphraser, opts, stats, translator)
                for (u_aug, o_aug, aug_tags) in enriched_variants:
                    rid_aug = f"{rid}-enriched{random.randint(1000,9999)}"
                    _commit_row(writer, source, rid_aug, "medical_dialogue", instr, u_aug, o_aug, opts, stats, aug_tags, dedupe_seen=dedupe_seen, translator=translator)
                
                # Add clinical scenarios for enhanced diversity
                if opts.get("clinical_scenarios", True):
                    clinical_scenarios = A.create_clinical_scenarios(user, out, paraphraser)
                    for (scenario_q, scenario_a, scenario_tag) in clinical_scenarios:
                        rid_scenario = f"{rid}-scenario{random.randint(1000,9999)}"
                        _commit_row(writer, source, rid_scenario, "medical_dialogue", instr, scenario_q, scenario_a, opts, stats, [scenario_tag], dedupe_seen=dedupe_seen, translator=translator)
                        stats["clinical_scenarios_created"] += 1
            
            # Increment count only on success
            count += 1
        except Exception as e:
            logger.warning(f"[PROC] {source} error in processing/augmentation for item {i}: {e}")
            continue
        if sample_limit and count >= sample_limit:
            break
        if cb and i % 1000 == 0:
            cb(min(0.9, 0.05 + i/200000), f"{source}: processed {i} rows")
    if cb:
        cb(0.92, f"{source} done ({count})")
    logger.info(f"[PROC] {source} done count={count} written={stats['written']} dedup_skipped={stats['dedup_skipped']}")
    return count

def _proc_pubmedqa_l(path, writer, paraphraser, opts, sample_limit, stats, cb, dedupe_seen=None, translator=None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    count = 0
    for k, v in data.items():
        try:
            q_raw = v.get("QUESTION") or ""
            ctx_list = v.get("CONTEXTS") or []
            long_ans_raw = v.get("LONG_ANSWER") or ""
            final_raw = v.get("final_decision") or ""
            
            # Ensure we have string values
            q = str(q_raw).strip() if q_raw else ""
            if isinstance(ctx_list, list):
                context = "\n".join(str(ctx) for ctx in ctx_list).strip()
            else:
                context = str(ctx_list).strip()
            long_ans = str(long_ans_raw).strip() if long_ans_raw else ""
            final = str(final_raw).strip() if final_raw else ""
        except Exception as e:
            logger.warning(f"[PROC] pubmedqa_l error processing item {k}: {e}, item: {v}")
            continue

        try:
            instr = "Answer the biomedical question using the provided context. Include a concise rationale if possible."
            user  = f"Question: {q}\n\nContext:\n{context}" if context else f"Question: {q}"
            out   = long_ans if long_ans else final
            rid   = str(k)

            instr, user, out, applied = _apply_aug(instr, user, out, "pubmedqa_l", opts, paraphraser, stats)
            
            # Skip if retry failed (empty output)
            if not out:
                stats["dropped_invalid"] = stats.get("dropped_invalid", 0) + 1
                continue
                
            _commit_row(writer, "pubmedqa_l", rid, "biomedical_qa", instr, user, out, opts, stats, applied,
                        extra_meta={"year": v.get("YEAR"), "meshes": v.get("MESHES"), "labels": v.get("LABELS")}, dedupe_seen=dedupe_seen, translator=translator)
            if opts.get("expand", True):
                # Use enriched variants for SFT (multiple Q&A combinations)
                enriched_variants = _build_enriched_variants(user, out, paraphraser, opts, stats, translator)
                for (u_aug, o_aug, aug_tags) in enriched_variants:
                    rid_aug = f"{rid}-enriched{random.randint(1000,9999)}"
                    _commit_row(writer, "pubmedqa_l", rid_aug, "biomedical_qa",
                                instr, u_aug, o_aug, opts, stats, aug_tags, dedupe_seen=dedupe_seen, translator=translator)

            # Increment count only on success
            count += 1
        except Exception as e:
            logger.warning(f"[PROC] pubmedqa_l error in processing/augmentation for item {k}: {e}")
            continue
        if sample_limit and count >= sample_limit:
            break
        if cb and count % 1000 == 0:
            cb(min(0.9, 0.05 + count/60000), f"pubmedqa_l processed {count}")
    if cb:
        cb(0.93, f"pubmedqa_l done ({count})")
    logger.info(f"[PROC] pubmedqa_l done count={count} written={stats['written']} dedup_skipped={stats['dedup_skipped']}")
    return count

def _proc_pubmedqa_u(path, writer, paraphraser, opts, sample_limit, stats, cb, dedupe_seen=None, translator=None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    count = 0
    for k, v in data.items():
        try:
            q_raw = v.get("QUESTION") or ""
            ctx_list = v.get("CONTEXTS") or []
            
            # Ensure we have string values
            q = str(q_raw).strip() if q_raw else ""
            if isinstance(ctx_list, list):
                context = "\n".join(str(ctx) for ctx in ctx_list).strip()
            else:
                context = str(ctx_list).strip()
        except Exception as e:
            logger.warning(f"[PROC] pubmedqa_u error processing item {k}: {e}, item: {v}")
            continue

        try:
            instr = "Rewrite the context into a succinct note, then answer the question. If unknown, say 'insufficient evidence'."
            user  = f"Question: {q}\n\nContext:\n{context}" if context else f"Question: {q}"
            out   = ""  # unlabeled
            rid   = str(k)

            # Optional KD/distillation for a small fraction
            if opts.get("distill_fraction", 0.0) > 0.0 and random.random() < float(opts["distill_fraction"]):
                prompt = f"{instr}\n\n{user}\n\nAnswer briefly and safely."
                guess = paraphraser.paraphrase(prompt, difficulty="hard")  # cheap single call
                if guess and len(guess) < 2000:
                    out = guess.strip()

            instr, user, out, applied = _apply_aug(instr, user, out, "pubmedqa_u", opts, paraphraser, stats)
            
            # Skip if retry failed (empty output)
            if not out:
                stats["dropped_invalid"] = stats.get("dropped_invalid", 0) + 1
                continue
                
            _commit_row(writer, "pubmedqa_u", str(k), "biomedical_qa_unlabeled", instr, user, out, opts, stats, applied, dedupe_seen=dedupe_seen, translator=translator)
            if opts.get("expand", True):
                # Use enriched variants for SFT (multiple Q&A combinations)
                enriched_variants = _build_enriched_variants(user, out, paraphraser, opts, stats, translator)
                for (u_aug, o_aug, aug_tags) in enriched_variants:
                    rid_aug = f"{rid}-enriched{random.randint(1000,9999)}"
                    _commit_row(writer, "pubmedqa_u", rid_aug, "biomedical_qa",
                                instr, u_aug, o_aug, opts, stats, aug_tags, dedupe_seen=dedupe_seen, translator=translator)
            
            # Increment count only on success
            count += 1
        except Exception as e:
            logger.warning(f"[PROC] pubmedqa_u error in processing/augmentation for item {k}: {e}")
            continue
        if sample_limit and count >= sample_limit:
            break
        if cb and count % 2000 == 0:
            cb(min(0.9, 0.05 + count/80000), f"pubmedqa_u processed {count}")
    if cb:
        cb(0.94, f"pubmedqa_u done ({count})")
    logger.info(f"[PROC] pubmedqa_u done count={count} written={stats['written']} dedup_skipped={stats['dedup_skipped']}")
    return count

def _proc_pubmedqa_map(path, writer, paraphraser, opts, sample_limit, stats, cb, dedupe_seen=None, translator=None):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    
    # Log the structure for debugging
    logger.info(f"[PROC] pubmedqa_map data type: {type(obj)}")
    if isinstance(obj, dict):
        logger.info(f"[PROC] pubmedqa_map dict keys: {list(obj.keys())}")
        if len(obj) > 0:
            sample_key = next(iter(obj.keys()))
            sample_value = obj[sample_key]
            logger.info(f"[PROC] pubmedqa_map sample value type: {type(sample_value)}")
            if isinstance(sample_value, dict):
                logger.info(f"[PROC] pubmedqa_map sample value keys: {list(sample_value.keys())}")
    
    # Iteration of items
    def iter_items():
        try:
            if isinstance(obj, list):
                for it in obj: 
                    if isinstance(it, dict):
                        yield it
                    else:
                        logger.warning(f"[PROC] pubmedqa_map skipping non-dict list item: {type(it)}")
            elif isinstance(obj, dict):
                qs, cs, ans = obj.get("question"), obj.get("context"), obj.get("answer")
                if isinstance(qs, list) and isinstance(cs, list) and isinstance(ans, list):
                    for i in range(min(len(qs), len(cs), len(ans))):
                        yield {"question": qs[i], "context": cs[i], "answer": ans[i]}
                else:
                    # Handle case where values might be dictionaries or other objects
                    for k, v in obj.items():
                        if isinstance(v, dict):
                            # If v is a dict, ensure it has the expected structure
                            if "question" in v and "context" in v and "answer" in v:
                                yield v
                            else:
                                # Try to map the keys to expected structure
                                yield {
                                    "question": v.get("question") or v.get("QUESTION") or str(k),
                                    "context": v.get("context") or v.get("CONTEXT") or "",
                                    "answer": v.get("answer") or v.get("ANSWER") or ""
                                }
                        else:
                            # If v is not a dict, create a simple structure
                            yield {"question": str(k), "context": str(v) if v else "", "answer": ""}
            else:
                logger.warning(f"[PROC] pubmedqa_map unexpected data type: {type(obj)}")
        except Exception as e:
            logger.error(f"[PROC] pubmedqa_map error in iter_items: {e}")
            return

    count = 0
    for i, v in enumerate(iter_items(), start=1):
        try:
            # Ensure we have string values, convert if necessary
            q_raw = v.get("question") or ""
            c_raw = v.get("context") or ""
            a_raw = v.get("answer") or ""
            
            # Convert to string if not already
            q = str(q_raw).strip() if q_raw else ""
            c = str(c_raw).strip() if c_raw else ""
            a = str(a_raw).strip() if a_raw else ""

            instr = "Answer the biomedical question based on the context. Justify briefly."
            user  = f"Question: {q}\n\nContext:\n{c}" if c else f"Question: {q}"
            out   = a
            rid   = _hash_id("pubmedqa_map", i, len(q))

            # Process the item
            instr, user, out, applied = _apply_aug(instr, user, out, "pubmedqa_map", opts, paraphraser, stats)
            
            # Skip if retry failed (empty output)
            if not out:
                stats["dropped_invalid"] = stats.get("dropped_invalid", 0) + 1
                continue
                
            _commit_row(writer, "pubmedqa_map", rid, "biomedical_qa", instr, user, out, opts, stats, applied, dedupe_seen=dedupe_seen, translator=translator)
            
            # Handle expansion if enabled
            if opts.get("expand", True):
                # Use enriched variants for SFT (multiple Q&A combinations)
                enriched_variants = _build_enriched_variants(user, out, paraphraser, opts, stats, translator)
                for (u_aug, o_aug, aug_tags) in enriched_variants:
                    rid_aug = f"{rid}-enriched{random.randint(1000,9999)}"
                    _commit_row(writer, "pubmedqa_map", rid_aug, "biomedical_qa",
                                instr, u_aug, o_aug, opts, stats, aug_tags, dedupe_seen=dedupe_seen, translator=translator)
            
            # Increment count only on success
            count += 1
            
        except Exception as e:
            logger.warning(f"[PROC] pubmedqa_map error processing item {i}: {e}, item: {v}")
            continue
            
        # Check sample limit
        if sample_limit and count >= sample_limit:
            break
        if cb and i % 2000 == 0:
            cb(min(0.9, 0.05 + i/120000), f"pubmedqa_map processed {i}")
    
    if cb:
        cb(0.95, f"pubmedqa_map done ({count})")
    logger.info(f"[PROC] pubmedqa_map done count={count} written={stats['written']} dedup_skipped={stats['dedup_skipped']}")
    return count
