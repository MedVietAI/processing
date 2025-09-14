# Centralized SFT writer (JSONL + CSV)
import csv
import orjson
from typing import Optional, Dict
import logging

# Logger
logger = logging.getLogger("schema")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

def sft_row(instruction: str, user_input: str, output: str, source: str, rid: str, task: str, meta: Optional[dict] = None):
    return {
        "source": source,
        "id": rid,
        "task": task,
        "sft": {
            "instruction": instruction,
            "input": user_input,
            "output": output
        },
        "meta": meta or {}
    }

def is_valid_row(row: Dict, max_chars: int = 20000) -> bool:
    s = row.get("sft", {})
    instr = s.get("instruction", "")
    inp = s.get("input", "")
    out = s.get("output", "")
    # basic sanity: non-empty input OR output; cap extremes
    if not (inp or out): return False
    if any(len(x) > max_chars for x in (instr, inp, out)): return False
    return True

class CentralisedWriter:
    """Streams JSONL + CSV in parallel to stay memory-safe."""
    def __init__(self, jsonl_path: str, csv_path: str):
        self.jsonl_fp = open(jsonl_path, "wb")
        self.csv_fp   = open(csv_path, "w", newline="", encoding="utf-8")
        self.csv_wr   = csv.DictWriter(self.csv_fp, fieldnames=["instruction","input","output","source","id","task"])
        self.csv_wr.writeheader()

    def write(self, row: dict):
        if not is_valid_row(row):
            s = row.get("sft", {})
            logger.warning(
                f"[WRITER] Skipping invalid row id={row.get('id')} "
                f"(len instr={len(s.get('instruction',''))}, input={len(s.get('input',''))}, output={len(s.get('output',''))})"
            )
            return
        self.jsonl_fp.write(orjson.dumps(row))
        self.jsonl_fp.write(b"\n")
        s = row["sft"]
        self.csv_wr.writerow({
            "instruction": s.get("instruction",""),
            "input": s.get("input",""),
            "output": s.get("output",""),
            "source": row.get("source",""),
            "id": row.get("id",""),
            "task": row.get("task","")
        })

    def close(self):
        try:
            self.jsonl_fp.close()
        finally:
            self.csv_fp.close()
