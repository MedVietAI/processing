# HF dataset download resolver + downloader
import os
from typing import Optional
from huggingface_hub import hf_hub_download
import logging

# Logger
logger = logging.getLogger("datasets")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())


DATASETS = {
    "healthcaremagic": {
        "repo_id":  "BinKhoaLe1812/MedDialog-EN-100k",
        "filename": "HealthCareMagic-100k.json",
        "repo_type": "dataset"
    },
    "icliniq": {
        "repo_id":  "BinKhoaLe1812/MedDialog-EN-10k",
        "filename": "iCliniq.json",
        "repo_type": "dataset"
    },
    "pubmedqa_l": {
        "repo_id":  "BinKhoaLe1812/PubMedQA-L",
        "filename": "ori_pqal.json",
        "repo_type": "dataset"
    },
    "pubmedqa_u": {
        "repo_id":  "BinKhoaLe1812/PubMedQA-U",
        "filename": "ori_pqau.json",
        "repo_type": "dataset"
    },
    "pubmedqa_map": {
        "repo_id":  "BinKhoaLe1812/PubMedQA-Map",
        "filename": "pubmed_qa_map.json",
        "repo_type": "dataset"
    }
}


def resolve_dataset(key: str) -> Optional[dict]:
    return DATASETS.get(key.lower())


def hf_download_dataset(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
    token = os.getenv("HF_TOKEN")
    logger.info(
        f"[HF] Download {repo_id}/{filename} (type={repo_type}) token={'yes' if token else 'no'}"
    )
    
    # Set cache directory with proper permissions
    cache_dir = os.path.abspath("cache/hf")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set HF_HOME to avoid permission issues
    hf_home = os.path.abspath("cache/huggingface")
    os.makedirs(hf_home, exist_ok=True)
    os.environ["HF_HOME"] = hf_home
    
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        token=token,
        local_dir=cache_dir,
        local_dir_use_symlinks=False
    )
    try:
        size = os.path.getsize(path)
        logger.info(f"[HF] Downloaded to {path} size={size} bytes")
    except Exception:
        logger.info(f"[HF] Downloaded to {path}")
    return path

