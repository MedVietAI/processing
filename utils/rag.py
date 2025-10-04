# RAG-specific dataset processor
import json
import logging
import hashlib
import random
from typing import Dict, List, Tuple, Optional, Callable

from utils.schema import sft_row, rag_row
from utils.llm import NvidiaClient, KeyRotator
from vi.processing import should_translate, translate_rag_row
from utils import augment as A

# Logger
logger = logging.getLogger("rag_processor")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

def _hash_id(*parts) -> str:
    """Generate a hash ID for RAG entries"""
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
    return h.hexdigest()[:16]

def _iter_json_or_jsonl(path: str):
    """Iterate over JSON or JSONL files"""
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            for obj in data:
                yield obj
        else:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

class RAGProcessor:
    """Processes medical datasets into RAG-specific QCA (Question, Context, Answer) format"""
    
    def __init__(self, nvidia_model: str):
        self.nvidia_client = NvidiaClient(KeyRotator("NVIDIA_API"), nvidia_model)
        
    def clean_conversational_content(self, text: str) -> str:
        """Remove conversational elements and non-medical information using NVIDIA model; keep concise for embeddings."""
        if not text or len(text.strip()) < 10:
            return text
            
        prompt = f"""
        You are a medical data cleaning expert. Clean the following text by:
        1. Remove conversational elements (greetings, pleasantries)
        2. Remove non-medical small talk and social interactions
        3. Keep only medically relevant information
        4. Preserve clinical facts, symptoms, diagnoses, treatments, and medical advice
        5. Maintain professional medical language
        6. Return only cleaned medical content in 1-2 concise sentences suitable for dense retrieval embeddings. No lists, no headers.

        Text to clean:
        {text}

        Cleaned medical content:"""

        try:
            cleaned = self.nvidia_client.generate(
                prompt, 
                temperature=0.1, 
                max_tokens=min(1000, len(text) + 200)
            )
            return cleaned.strip() if cleaned else text
        except Exception as e:
            logger.warning(f"[RAG] Error cleaning text: {e}")
            return text
    
    def generate_context_from_qa(self, question: str, answer: str) -> str:
        """Generate synthetic, concise context (<=2 sentences) from question and answer, embedding-friendly."""
        if not question or not answer:
            return ""
            
        prompt = f"""You are a medical knowledge expert. Given a medical question and its answer, generate a brief relevant medical context that helps retrieval. Limit to 1–2 sentences, concise, avoid boilerplate, no enumerations.

        Question: {question}

        Answer: {answer}

        Generate a concise medical context:"""

        try:
            context = self.nvidia_client.generate(
                prompt,
                temperature=0.2,
                max_tokens=200
            )
            # Trim to a single short paragraph
            return (context or "").strip().split("\n")[0][:600]
        except Exception as e:
            logger.warning(f"[RAG] Error generating context: {e}")
            return ""
    
    def convert_to_qca_format(self, instruction: str, user_input: str, output: str) -> Tuple[str, str, str]:
        """Convert SFT format to QCA (Question, Context, Answer) format, compressing for embedding suitability."""
        # Clean the content to remove conversational elements
        cleaned_input = self.clean_conversational_content(user_input)
        cleaned_output = self.clean_conversational_content(output)
        # Hard caps for embedding friendliness
        cleaned_input = (cleaned_input or "")[:1200]
        cleaned_output = (cleaned_output or "")[:1200]
        
        # Extract question from user input
        question = self.extract_question(cleaned_input)
        
        # Extract or generate context
        context = self.extract_context(cleaned_input, question, cleaned_output)
        
        # Clean answer
        # Prefer short, direct answers
        answer = cleaned_output[:800]
        
        return question, context, answer
    
    def extract_question(self, user_input: str) -> str:
        """Extract the main question from user input"""
        if not user_input:
            return ""
            
        # Try to identify question patterns
        lines = user_input.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Question:') or line.startswith('Q:'):
                return line.replace('Question:', '').replace('Q:', '').strip()
            elif '?' in line and len(line) > 10:
                return line
        
        # If no clear question found, use the first meaningful line
        for line in lines:
            line = line.strip()
            if len(line) > 10:
                return line
                
        return user_input
    
    def extract_context(self, user_input: str, question: str, answer: str) -> str:
        """Extract context from user input or generate synthetic context"""
        # Look for context in the original input
        context_candidates = []
        lines = user_input.split('\n')
        
        for line in lines:
            line = line.strip()
            if (line.startswith('Context:') or 
                line.startswith('Background:') or 
                line.startswith('Information:') or
                (len(line) > 50 and not line.startswith('Question:') and '?' not in line)):
                context_candidates.append(line)
        
        if context_candidates:
            # Clean and combine context candidates
            context = ' '.join(context_candidates)
            context = self.clean_conversational_content(context)
            if len(context) > 20:  # Ensure we have meaningful context
                return context
        
        # Generate synthetic context if none found
        if question and answer:
            synthetic_context = self.generate_context_from_qa(question, answer)
            if synthetic_context:
                return synthetic_context
        
        return ""
    
    def process_medical_dialog(self, source: str, path: str, writer, sample_limit: Optional[int], 
                             stats: Dict, progress_cb: Optional[Callable], dedupe_seen: set = None, translator=None, opts=None) -> int:
        """Process medical dialogue datasets into RAG format"""
        count = 0
        written = 0
        
        for i, obj in enumerate(_iter_json_or_jsonl(path), start=1):
            try:
                instr_raw = obj.get("instruction") or "Answer the medical question based on the provided context."
                user_raw = obj.get("input") or ""
                out_raw = obj.get("output") or ""
                
                instr = str(instr_raw).strip()
                user = str(user_raw).strip()
                out = str(out_raw).strip()
                rid = _hash_id(source, i, len(user), len(out))
                
                # Convert to QCA format
                question, context, answer = self.convert_to_qca_format(instr, user, out)
                
                # Clean invalid responses with retry logic
                if A.is_invalid_response(answer):
                    if paraphraser:
                        answer = A.retry_invalid_response(answer, paraphraser, max_retries=3)
                    else:
                        answer = A.clean_invalid_response(answer, "")
                    if not answer:  # If retry failed, skip this sample
                        continue
                
                if not question or not answer:
                    continue
                
                # Commit the RAG-formatted row (QAC)
                if self._commit_rag_row(writer, rid, question, context, answer,
                                      stats, dedupe_seen=dedupe_seen, translator=translator, opts=opts):
                    written += 1
                
                count += 1
                
            except Exception as e:
                logger.warning(f"[RAG] {source} error processing item {i}: {e}")
                continue
                
            if sample_limit and count >= sample_limit:
                break
            if progress_cb and i % 1000 == 0:
                progress_cb(min(0.9, 0.05 + i/200000), f"{source}: processed {i} rows for RAG")
        
        if progress_cb:
            progress_cb(0.92, f"{source} RAG processing done ({count})")
        
        logger.info(f"[RAG] {source} RAG processing done count={count} written={written}")
        return count
    
    def process_pubmedqa(self, source: str, path: str, writer, sample_limit: Optional[int], 
                        stats: Dict, progress_cb: Optional[Callable], dedupe_seen: set = None, translator=None, opts=None) -> int:
        """Process PubMedQA datasets into RAG format"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        count = 0
        written = 0
        
        for k, v in data.items():
            try:
                q_raw = v.get("QUESTION") or ""
                ctx_list = v.get("CONTEXTS") or []
                long_ans_raw = v.get("LONG_ANSWER") or ""
                final_raw = v.get("final_decision") or ""
                
                question = str(q_raw).strip() if q_raw else ""
                if isinstance(ctx_list, list):
                    context = "\n".join(str(ctx) for ctx in ctx_list).strip()
                else:
                    context = str(ctx_list).strip()
                answer = str(long_ans_raw).strip() if long_ans_raw else str(final_raw).strip()
                
                if not question or not answer:
                    continue
                
                # Clean the content
                question = self.clean_conversational_content(question)
                context = self.clean_conversational_content(context)
                answer = self.clean_conversational_content(answer)
                
                # Clean invalid responses with retry logic
                if A.is_invalid_response(answer):
                    if paraphraser:
                        answer = A.retry_invalid_response(answer, paraphraser, max_retries=3)
                    else:
                        answer = A.clean_invalid_response(answer, "")
                    if not answer:  # If retry failed, skip this sample
                        continue
                
                # Generate context if missing
                if not context:
                    context = self.generate_context_from_qa(question, answer)
                
                rid = str(k)
                # Commit the RAG-formatted row (QAC)
                if self._commit_rag_row(writer, rid, question, context, answer,
                                      stats, dedupe_seen=dedupe_seen, translator=translator, opts=opts):
                    written += 1
                
                count += 1
                
            except Exception as e:
                logger.warning(f"[RAG] {source} error processing item {k}: {e}")
                continue
                
            if sample_limit and count >= sample_limit:
                break
            if progress_cb and count % 1000 == 0:
                progress_cb(min(0.9, 0.05 + count/60000), f"{source} RAG processed {count}")
        
        if progress_cb:
            progress_cb(0.93, f"{source} RAG processing done ({count})")
        
        logger.info(f"[RAG] {source} RAG processing done count={count} written={written}")
        return count
    
    def _commit_rag_row(self, writer, rid: str, question: str, context: str, answer: str, 
                       stats: Dict, dedupe_seen: set = None, translator=None, opts=None) -> bool:
        """Commit a RAG-formatted row (QAC) to the writer"""
        # Simple deduplication based on content hash
        if dedupe_seen is not None:
            content_hash = hashlib.md5(f"{question}{context}{answer}".encode()).hexdigest()
            if content_hash in dedupe_seen:
                stats["dedup_skipped"] = stats.get("dedup_skipped", 0) + 1
                return False
            dedupe_seen.add(content_hash)

        row = rag_row(question=question, context=context, answer=answer, rid=rid)

        # Apply Vietnamese translation if requested (translate Q/A/C fields directly)
        if should_translate(opts.get("vietnamese_translation", False) if opts else False, translator):
            try:
                row = translate_rag_row(row, translator, ["question", "answer", "context"])
                # Add translation metadata
                if "meta" not in row:
                    row["meta"] = {}
                row["meta"]["vietnamese_translated"] = True
            except Exception as e:
                logger.error(f"Failed to translate RAG row: {e}")
                # Continue with original row if translation fails

        writer.write(row)
        stats["written"] = stats.get("written", 0) + 1
        return True

def process_file_into_rag(
    dataset_key: str,
    input_path: str,
    writer,
    nvidia_model: str,
    sample_limit: Optional[int],
    seed: int,
    progress_cb: Optional[Callable[[float, str], None]],
    translator=None,
    paraphraser=None
) -> Tuple[int, Dict]:
    """Main entry point for RAG processing"""
    random.seed(seed)
    stats = {
        "written": 0,
        "dedup_skipped": 0
    }
    
    logger.info(f"[RAG] Begin RAG processing dataset={dataset_key} sample_limit={sample_limit}")
    
    # Initialize RAG processor
    rag_processor = RAGProcessor(nvidia_model)
    dedupe_seen = set()
    
    key = dataset_key.lower()
    # Create opts with Vietnamese translation flag
    opts = {"vietnamese_translation": translator is not None}
    
    if key in ("healthcaremagic", "icliniq"):
        count = rag_processor.process_medical_dialog(
            source=key, path=input_path, writer=writer,
            sample_limit=sample_limit, stats=stats, 
            progress_cb=progress_cb, dedupe_seen=dedupe_seen, translator=translator, opts=opts
        )
    elif key in ("pubmedqa_l", "pubmedqa_u", "pubmedqa_map"):
        count = rag_processor.process_pubmedqa(
            source=key, path=input_path, writer=writer,
            sample_limit=sample_limit, stats=stats, 
            progress_cb=progress_cb, dedupe_seen=dedupe_seen, translator=translator, opts=opts
        )
    else:
        raise ValueError(f"Unknown dataset for RAG processing: {dataset_key}")
    
    logger.info(f"[RAG] End RAG processing dataset={dataset_key} stats={stats}")
    return count, stats