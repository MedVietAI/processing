# Root FastAPI
import os
import json
import time, logging
import threading
import datetime as dt
from typing import Optional, Dict

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request 
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from utils.datasets import resolve_dataset, hf_download_dataset
from utils.processor import process_file_into_sft
from utils.rag import process_file_into_rag
from utils.drive_saver import DriveSaver
from utils.llm import Paraphraser
from utils.schema import CentralisedWriter, RAGWriter
from utils.token import get_credentials, exchange_code, build_auth_url
from vi.translator import VietnameseTranslator

# ────────── Log ───────────
logger = logging.getLogger("app")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

# ────────── Boot ──────────
load_dotenv(override=True)

SPACE_NAME = os.getenv("SPACE_NAME", "MedAI Processor")
OUTPUT_DIR = os.path.abspath(os.getenv("OUTPUT_DIR", "cache/outputs"))
LOG_DIR = os.path.abspath(os.getenv("LOG_DIR", "logs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Bootstrap Google OAuth ---
try:
    creds = get_credentials()
    if creds:
        logger.info("✅ OAuth credentials loaded and valid")
except Exception as e:
    logger.warning(f"⚠️ OAuth not initialized yet: {e}")

# --- Bootstrap Google Drive ---
drive = DriveSaver(default_folder_id=os.getenv("GDRIVE_FOLDER_ID"))

# LLM rotator with paraphraser nodes
paraphraser = Paraphraser(
    nvidia_model=os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct"),
    gemini_model_easy=os.getenv("GEMINI_MODEL_EASY", "gemini-2.5-flash-lite"),
    gemini_model_hard=os.getenv("GEMINI_MODEL_HARD", "gemini-2.5-flash"),
)

# Vietnamese translator (currently using Helsinki-NLP/opus-mt-en-vi)
vietnamese_translator = VietnameseTranslator()

app = FastAPI(title="Medical Dataset Augmenter", version="1.1.0")

STATE_LOCK = threading.Lock()
STATE: Dict[str, object] = {
    "running": False,
    "dataset": None,
    "started_at": None,
    "progress": 0.0,
    "message": "idle",
    "last_result": None
}

class AugmentOptions(BaseModel):
    # ratios are 0..1
    paraphrase_ratio: float = 0.2
    paraphrase_outputs: bool = True
    backtranslate_ratio: float = 0.1
    style_standardize: bool = True
    deidentify: bool = True
    dedupe: bool = True
    max_chars: int = 5000                 # cap extremely long contexts
    consistency_check_ratio: float = 0.05  # small ratio e.g. 0.01
    # KD / distillation (optional, keeps default off)
    distill_fraction: float = 0.0         # for unlabeled only
    expand: bool = True                   # Enable back-translation and complex augmentation
    max_aug_per_sample: int = 2           # Between 1-3, number of LLM call to augment/paraphrase data

class ProcessParams(BaseModel):
    augment: AugmentOptions = AugmentOptions()
    sample_limit: Optional[int] = None    # Set data sampling if needed 
    seed: int = 42
    rag_processing: bool = False          # Enable RAG-specific processing
    vietnamese_translation: bool = False  # Enable Vietnamese translation

def set_state(**kwargs):
    with STATE_LOCK:
        STATE.update(kwargs)

def now_iso():
    return dt.datetime.utcnow().isoformat()

# Instructional UI
@app.get("/", response_class=HTMLResponse)
def root():
    return f"""
    <html>
    <head>
      <title>{SPACE_NAME} – Medical Dataset Augmenter</title>
      <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 2rem auto; line-height: 1.5; }}
        h1, h2 {{ color: #2c3e50; }}
        button {{
          background: #2d89ef; color: white; border: none; padding: 8px 16px;
          border-radius: 5px; cursor: pointer; margin: 5px 0;
        }}
        button:hover {{ background: #1b5dab; }}
        .section {{ margin-bottom: 2rem; }}
        #log {{ background:#f5f5f5; padding:10px; border-radius:6px; margin-top:10px; font-size:0.9rem; }}
        a {{ color:#2d89ef; text-decoration:none; }}
        a:hover {{ text-decoration:underline; }}
      </style>
    </head>
    <body>
      <h1>📊 {SPACE_NAME} – Medical Dataset Augmenter</h1>
      <p>This Hugging Face Space processes medical datasets into a <b>centralised fine-tuning format</b>
         (JSONL + CSV), with optional <i>data augmentation</i>.</p>

      <div class="section">
        <h2>⚡ Quick Actions</h2>
        <p>Click a button below to start processing a dataset with default augmentation parameters.</p>
        
        <div style="margin-bottom: 15px; padding: 10px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #2d89ef;">
          <label style="display: flex; align-items: center; cursor: pointer;">
            <input type="checkbox" id="vietnameseTranslation" style="margin-right: 8px; transform: scale(1.2);">
            <strong>🇻🇳 Vietnamese Translation</strong> - Translate all content to Vietnamese before processing
          </label>
        </div>
        
        <button onclick="startJob('healthcaremagic')">▶ProcAugment HealthCareMagic (100k)</button><br>
        <button onclick="startJob('icliniq')">▶ProcAugment iCliniq (10k-derived)</button><br>
        <button onclick="startJob('pubmedqa_l')">▶ProcAugment PubMedQA (Labelled)</button><br>
        <button onclick="startJob('pubmedqa_u')">▶ProcAugment PubMedQA (Unlabelled)</button><br>
        <button onclick="startJob('pubmedqa_map')">▶ProcAugment PubMedQA (Map)</button><br><br>
        <div style="border-top: 1px solid #ddd; padding-top: 10px; margin-top: 10px;">
          <strong>RAG Processing:</strong> - Convert to QCA format for RAG systems<br>
          <button onclick="startRagJob('healthcaremagic')" style="background: #e74c3c;">▶ RAG HealthCareMagic (100k)</button><br>
          <button onclick="startRagJob('icliniq')" style="background: #e74c3c;">▶ RAG iCliniq (10k-derived)</button><br>
          <button onclick="startRagJob('pubmedqa_u')" style="background: #e74c3c;">▶ RAG PubMedQA (Unlabelled)</button><br>
          <button onclick="startRagJob('pubmedqa_l')" style="background: #e74c3c;">▶ RAG PubMedQA (Labelled)</button><br>
          <button onclick="startRagJob('pubmedqa_map')" style="background: #e74c3c;">▶ RAG PubMedQA (Map)</button>
        </div>
      </div>

      <div class="section">
        <h2>📂 Monitoring</h2>
        <ul>
          <li><a href="/status" target="_blank">Check current job status</a></li>
          <li><a href="/files" target="_blank">List generated artifacts</a></li>
          <li><a href="https://medvietai-processing.hf.space/oauth2/start" target="_blank">Authorize your GCS credential</a></li>
          <li><a href="https://huggingface.co/spaces/BinKhoaLe1812/MedAI_Processing/blob/main/REQUEST.md" target="_blank">📑 Request Doc (all curl examples)</a></li>
        </ul>
      </div>

      <div class="section">
        <h2>📝 Log</h2>
        <div id="log">Click a button above to run a job...</div>
      </div>

      <script>
        async function startJob(dataset) {{
          const log = document.getElementById("log");
          const vietnameseToggle = document.getElementById("vietnameseTranslation");
          const isVietnameseMode = vietnameseToggle.checked;
          
          log.innerHTML = "⏳ Starting job for <b>" + dataset + "</b>" + (isVietnameseMode ? " with Vietnamese translation" : "") + "...";
          try {{
            const resp = await fetch("/process/" + dataset, {{
              method: "POST",
              headers: {{ "Content-Type": "application/json" }},
              body: JSON.stringify({{
                augment: {{
                  paraphrase_ratio: 0.2,
                  backtranslate_ratio: 0.1,
                  paraphrase_outputs: true,
                  style_standardize: true,
                  deidentify: true,
                  dedupe: true,
                  max_chars: 5000,
                  expand: true,
                  max_aug_per_sample: 2,
                  consistency_check_ratio: 0.05
                }},
                sample_limit: null,          // Sample down (currently disabled)
                seed: 42,
                rag_processing: false,
                vietnamese_translation: isVietnameseMode
              }})
            }});
            const data = await resp.json();
            if (resp.ok) {{
              log.innerHTML = "✅ " + JSON.stringify(data);
            }} else {{
              log.innerHTML = "❌ Error: " + JSON.stringify(data);
            }}
          }} catch (err) {{
            log.innerHTML = "❌ JS Error: " + err;
          }}
        }}
        
        async function startRagJob(dataset) {{
          const log = document.getElementById("log");
          const vietnameseToggle = document.getElementById("vietnameseTranslation");
          const isVietnameseMode = vietnameseToggle.checked;
          
          log.innerHTML = "⏳ Starting RAG processing for <b>" + dataset + "</b>" + (isVietnameseMode ? " with Vietnamese translation" : "") + "...";
          try {{
            const resp = await fetch("/rag/" + dataset, {{
              method: "POST",
              headers: {{ "Content-Type": "application/json" }},
              body: JSON.stringify({{
                sample_limit: null,
                seed: 42,
                vietnamese_translation: isVietnameseMode
              }})
            }});
            const data = await resp.json();
            if (resp.ok) {{
              log.innerHTML = "✅ RAG Processing Started: " + JSON.stringify(data);
            }} else {{
              log.innerHTML = "❌ Error: " + JSON.stringify(data);
            }}
          }} catch (err) {{
            log.innerHTML = "❌ JS Error: " + err;
          }}
        }}
      </script>
    </body>
    </html>
    """

@app.get("/status")
def status():
    with STATE_LOCK:
        return JSONResponse(STATE)
    
# ──────── GCS token ────────
@app.get("/oauth2/start")
def oauth2_start(request: Request):
    # Compute redirect URI dynamically from the actual host the Space is using
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    scheme = "https"  # Spaces are HTTPS at the edge
    redirect_uri = f"{scheme}://{host}/oauth2/callback"

    try:
        url = build_auth_url(redirect_uri)
        return JSONResponse({"authorize_url": url})
    except Exception as e:
        raise HTTPException(500, f"OAuth init failed: {e}")

# Display your token
@app.get("/oauth2/callback")
def oauth2_callback(request: Request, code: str = "", state: str = ""):
    if not code:
        raise HTTPException(400, "Missing 'code'")
    # Send req
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    scheme = "https"
    redirect_uri = f"{scheme}://{host}/oauth2/callback"
    # Parse and show token code
    try:
        creds = exchange_code(code, redirect_uri)
        refresh = creds.refresh_token or os.getenv("GDRIVE_REFRESH_TOKEN", "")
        # UI
        html = f"""
        <html>
        <head>
          <style>
            body {{ font-family: sans-serif; margin: 2em; }}
            .token-box {{
              padding: 1em; border: 1px solid #ccc; border-radius: 6px;
              background: #f9f9f9; font-family: monospace;
              word-break: break-all; white-space: pre-wrap;
            }}
            .note {{ margin-top: 1em; color: #555; }}
          </style>
        </head>
        <body>
          <h2>✅ Google Drive Authorized</h2>
          <p>Your refresh token is:</p>
          <div class="token-box">{refresh}</div>
          <p class="note">
            👉 Copy this token and save it into your Hugging Face Space Secrets
            as <code>GDRIVE_REFRESH_TOKEN</code>.  
            This ensures persistence across rebuilds.
          </p>
        </body>
        </html>
        """
        return HTMLResponse(html)
    except Exception as e:
        raise HTTPException(500, f"OAuth exchange failed: {e}")

@app.get("/files")
def files():
    out = []
    for root, _, fns in os.walk(OUTPUT_DIR):
        for fn in fns:
            out.append(os.path.relpath(os.path.join(root, fn), OUTPUT_DIR))
    return {"output_dir": OUTPUT_DIR, "files": sorted(out)}

@app.post("/process/{dataset_key}")
def process_dataset(dataset_key: str, params: ProcessParams, background: BackgroundTasks):
    with STATE_LOCK:
        if STATE["running"]:
            logger.warning(
                f"[JOB] Rejecting new job dataset={dataset_key} "
                f"current={STATE['dataset']} started_at={STATE['started_at']}"
            )
            raise HTTPException(409, detail="Another job is running.")
        STATE["running"] = True
        STATE["dataset"] = dataset_key
        STATE["started_at"] = now_iso()
        STATE["progress"] = 0.0
        STATE["message"] = "starting"
        STATE["last_result"] = None
        logger.info(
            f"[JOB] Queued dataset={dataset_key} "
            f"params={{'sample_limit': {params.sample_limit}, 'seed': {params.seed}, "
            f"'rag_processing': {params.rag_processing}, 'augment': {params.augment.dict()} }}"
        )
    # Start job to background runner thread
    logger.info(f"[JOB] Started dataset={dataset_key}")
    background.add_task(_run_job, dataset_key, params)
    return {"ok": True, "message": f"Job for '{dataset_key}' started."}

@app.post("/rag/{dataset_key}")
def process_rag_dataset(dataset_key: str, params: ProcessParams, background: BackgroundTasks):
    """Dedicated RAG processing endpoint"""
    # Force RAG processing mode
    params.rag_processing = True
    
    with STATE_LOCK:
        if STATE["running"]:
            logger.warning(
                f"[RAG] Rejecting new RAG job dataset={dataset_key} "
                f"current={STATE['dataset']} started_at={STATE['started_at']}"
            )
            raise HTTPException(409, detail="Another job is running.")
        STATE["running"] = True
        STATE["dataset"] = dataset_key
        STATE["started_at"] = now_iso()
        STATE["progress"] = 0.0
        STATE["message"] = "starting RAG processing"
        STATE["last_result"] = None
        logger.info(
            f"[RAG] Queued RAG dataset={dataset_key} "
            f"params={{'sample_limit': {params.sample_limit}, 'seed': {params.seed} }}"
        )
    # Start job to background runner thread
    logger.info(f"[RAG] Started RAG dataset={dataset_key}")
    background.add_task(_run_job, dataset_key, params)
    return {"ok": True, "message": f"RAG processing job for '{dataset_key}' started."}

def _run_job(dataset_key: str, params: ProcessParams):
    t0 = time.time()
    try:
        ds = resolve_dataset(dataset_key)
        if not ds:
            set_state(running=False, message="unknown dataset")
            return
        
        # Download HF Dataset and start processing units
        set_state(message="downloading")
        local_path = hf_download_dataset(ds["repo_id"], ds["filename"], ds["repo_type"])
        logger.info(f"[JOB] Downloaded {ds['repo_id']}/{ds['filename']} → {local_path}")

        # Prepare timestamp for fire writing
        ts = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        mode_suffix = "rag" if params.rag_processing else "sft"
        stem = f"{dataset_key}-{mode_suffix}-{ts}"
        jsonl_path = os.path.join(OUTPUT_DIR, f"{stem}.jsonl")
        csv_path   = os.path.join(OUTPUT_DIR, f"{stem}.csv")
        # Change state
        set_state(message="processing", progress=0.05)

        # Writer
        writer = RAGWriter(jsonl_path=jsonl_path, csv_path=csv_path) if params.rag_processing else CentralisedWriter(jsonl_path=jsonl_path, csv_path=csv_path)
        
        # Load translator if Vietnamese translation is requested
        translator = None
        if params.vietnamese_translation:
            set_state(message="Loading Vietnamese translator", progress=0.05)
            try:
                # Ensure cache directories are set up properly
                cache_dir = os.path.abspath("cache/huggingface")
                os.makedirs(cache_dir, exist_ok=True)
                os.environ["HF_HOME"] = cache_dir

                # Pass paraphraser to translator for LLM-based translation
                vietnamese_translator.paraphraser = paraphraser
                vietnamese_translator.load_model()
                translator = vietnamese_translator
                logger.info("✅ Vietnamese translator loaded successfully with LLM models")
            except Exception as e:
                logger.error(f"❌ Failed to load Vietnamese translator: {e}")
                logger.warning("Continuing without Vietnamese translation...")
                set_state(message=f"Warning: Vietnamese translation disabled - {e}", progress=0.1)
                # Don't fail the entire job, just disable translation
                translator = None
        
        if params.rag_processing:
            # RAG processing mode
            set_state(message="RAG processing", progress=0.1)
            count, stats = process_file_into_rag(
                dataset_key=dataset_key,
                input_path=local_path,
                writer=writer,
                nvidia_model=os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct"),
                sample_limit=params.sample_limit,
                seed=params.seed,
                progress_cb=lambda p, msg=None: set_state(progress=p, message=msg or STATE["message"]),
                translator=translator,
                paraphraser=paraphraser
            )
        else:
            # Standard SFT processing mode
            set_state(message="SFT processing", progress=0.1)
            # Add Vietnamese translation flag to augment options
            augment_opts = params.augment.dict()
            augment_opts["vietnamese_translation"] = params.vietnamese_translation
            
            count, stats = process_file_into_sft(
                dataset_key=dataset_key,
                input_path=local_path,
                writer=writer,
                paraphraser=paraphraser,
                augment_opts=augment_opts,
                sample_limit=params.sample_limit,
                seed=params.seed,
                progress_cb=lambda p, msg=None: set_state(progress=p, message=msg or STATE["message"]),
                translator=translator
            )
        # Log translation statistics if translator was used
        if translator and hasattr(translator, 'get_stats'):
            translation_stats = translator.get_stats()
            logger.info(f"[JOB] Translation stats: {translation_stats}")
            stats["translation_stats"] = translation_stats
        
        logger.info(f"[JOB] Processed dataset={dataset_key} rows={count} stats={stats}")
        writer.close()

        # Upload to GDrive
        set_state(message="uploading to Google Drive", progress=0.95)
        up1 = drive.upload_file_to_drive(jsonl_path, mimetype="application/json")
        up2 = drive.upload_file_to_drive(csv_path,   mimetype="text/csv")
        logger.info(
            f"[JOB] Uploads complete uploaded={bool(up1 and up2)} "
            f"jsonl={jsonl_path} csv={csv_path}"
        )
        
        # Finalize a task
        result = {
            "dataset": dataset_key,
            "processing_mode": "RAG" if params.rag_processing else "SFT",
            "processed_rows": count,
            "stats": stats,
            "artifacts": {"jsonl": jsonl_path, "csv": csv_path},
            "uploaded": bool(up1 and up2),
            "duration_sec": round(time.time() - t0, 2)
        }
        set_state(message="done", progress=1.0, last_result=result, running=False)
        logger.info(
            f"[JOB] Finished dataset={dataset_key} "
            f"duration_sec={round(time.time()-t0, 2)}"
        )
    except Exception as e:
        logger.exception(f"[JOB] Error for dataset={dataset_key}: {e}")
        set_state(message=f"error: {e}", running=False)
