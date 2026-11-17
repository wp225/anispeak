#!/usr/bin/env python3
"""
filter_snapshot_bert_stream_v4.py
======================================
Producer-consumer streaming pipeline with multi-process parallelism.

Changes from v3:
  - Multi-GPU support with per-GPU config (batch_size, chunk_size, queue_max, num_readers)
  - File-level resume: completed .gz files are skipped entirely on restart
  - Completed-files manifest keyed by FILENAME (not worker_id) — safe to change
    gpu_configs between runs without losing progress
  - Seed-stable file partitioning: same seed + same file list = same assignment
  - Writer has 5-min timeout so it never hangs if a GPU worker is hard-killed
  - On --resume, pass1_bert.jsonl is backed up automatically before appending
  - Backup/merge utility modes baked in (--backup-only, --merge-only)

RUN (two GPUs, different memory budgets):
    CUDA_VISIBLE_DEVICES=0,1 python -u filter_snapshot_bert_stream_v4.py \\
      --snapshot_dir /path/to/openalex-works \\
      --output_dir   /path/to/corpus_output \\
      --model        /path/to/hf_model \\
      --config       bert_classifier_config.yaml \\
      --gpu_configs  "0:batch=1024,chunk=16384,queue=3,readers=2" \\
                     "1:batch=2048,chunk=16384,queue=4,readers=3" \\
      --threshold    0.45 \\
      --year_start   2000 \\
      --year_end     2026 \\
      --seed         42 \\
      --resume

RUN (single GPU):
    CUDA_VISIBLE_DEVICES=0 python -u filter_snapshot_bert_stream_v4.py \\
      --snapshot_dir /path/to/openalex-works \\
      --output_dir   /path/to/corpus_output \\
      --model        /path/to/hf_model \\
      --config       bert_classifier_config.yaml \\
      --gpu_configs  "0:batch=2048,chunk=16384,queue=4,readers=2" \\
      --threshold    0.45 \\
      --resume

BACKUP + MERGE ONLY (run before killing a live pipeline):
    python -u filter_snapshot_bert_stream_v4.py \\
      --output_dir /path/to/corpus_output \\
      --backup-only

MERGE DEDUP ONLY:
    python -u filter_snapshot_bert_stream_v4.py \\
      --output_dir /path/to/corpus_output \\
      --merge-only

NOTE ON GPU CONFIG CHANGES BETWEEN RUNS:
    You can freely change --gpu_configs and --num_readers between runs.
    The completed-files manifest is keyed by the .gz FILENAME (not worker index),
    so file-level skipping survives any config change. The dedup.tsv also survives
    any config change. Only ensure --seed stays the same so file assignment is
    reproducible (prevents two workers being assigned the same file if you add GPUs).
"""

import argparse
import os
import sys
import time
import shutil
import signal
import ctypes
import traceback
import subprocess
import multiprocessing as mp
from multiprocessing import Process, Queue, Value
from datetime import datetime
from glob import glob
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Dict

import torch
import numpy as np
import yaml
import orjson
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import build_text


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

KEEP_FIELDS = frozenset({
    "id", "doi", "display_name", "title",
    "publication_year", "publication_date",
    "type", "language",
    "primary_location", "locations", "best_oa_location",
    "open_access", "authorships",
    "topics", "concepts", "keywords",
    "abstract_inverted_index",
    "biblio", "cited_by_count", "counts_by_year",
    "is_retracted", "is_paratext",
    "referenced_works", "related_works",
    "grants", "mesh", "primary_topic",
    "indexed_in", "updated_date", "created_date",
})

ALLOWED_TYPES = frozenset({
    "article",              # standard peer-reviewed journal article
    "preprint",             # arXiv / bioRxiv / ESSOAr / etc.
    "book",                 # full monographs & textbooks
    "book-chapter",         # chapters in edited volumes
    "dataset",              # published data papers / repositories
    "dissertation",         # PhD / master's theses
    "review",               # systematic / narrative review articles
    "proceedings-article",  # ACL, NeurIPS, ICML, etc.
    "report",               # technical reports
    "standard",             # ISO / IEEE standards
    "other",                # catch-all — BERT will filter irrelevant ones
})

DEDUP_SYNC_INTERVAL    = 100_000
COUNTER_FLUSH_INTERVAL = 1_000

# Writer will give up waiting for GPU output after this many seconds.
# Protects against the writer hanging forever if a GPU worker is hard-killed.
WRITER_QUEUE_TIMEOUT_S = 1500

# Single shared completed-files manifest in output_dir (basename-keyed)
COMPLETED_MANIFEST_NAME = "completed_files.txt"


# ─────────────────────────────────────────────────────────────────────────────
# PER-GPU CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GpuConfig:
    """Configuration for one GPU worker and its dedicated reader pool."""
    device_id:   int = 0
    batch_size:  int = 2048
    chunk_size:  int = 16384
    queue_max:   int = 4
    num_readers: int = 2

    @property
    def device(self) -> str:
        return f"cuda:{self.device_id}"

    def __str__(self):
        return (f"cuda:{self.device_id} | batch={self.batch_size} "
                f"chunk={self.chunk_size} queue={self.queue_max} "
                f"readers={self.num_readers}")


def parse_gpu_config(spec: str) -> GpuConfig:
    """
    Parse a GPU config string: "DEVICE_ID:batch=N,chunk=N,queue=N,readers=N"
    All key=value pairs are optional; GpuConfig defaults fill missing ones.
    Examples:
        "0"                                          -> cuda:0 all defaults
        "0:batch=1024"                               -> cuda:0, batch=1024
        "1:batch=2048,chunk=16384,queue=4,readers=3"
    """
    if ":" not in spec:
        return GpuConfig(device_id=int(spec.strip()))
    dev_str, rest = spec.split(":", 1)
    cfg = GpuConfig(device_id=int(dev_str.strip()))
    for kv in rest.split(","):
        kv = kv.strip()
        if not kv:
            continue
        if "=" not in kv:
            raise ValueError(f"Malformed key=value in GPU spec: {kv!r} (from {spec!r})")
        k, v = kv.split("=", 1)
        k, v = k.strip(), int(v.strip())
        if   k == "batch":   cfg.batch_size  = v
        elif k == "chunk":   cfg.chunk_size  = v
        elif k == "queue":   cfg.queue_max   = v
        elif k == "readers": cfg.num_readers = v
        else:
            raise ValueError(f"Unknown GPU config key: {k!r} in {spec!r}")
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# DEDUP HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_dedup_tsv(tsv_path: str) -> dict:
    dedup = {}
    if not os.path.exists(tsv_path):
        return dedup
    with open(tsv_path, "r", buffering=64 * 1024 * 1024) as f:
        next(f, None)  # skip header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            tab = line.rfind("\t")
            if tab == -1:
                continue
            try:
                dedup[line[:tab]] = int(line[tab + 1:])
            except ValueError:
                continue
    return dedup


def save_dedup_tsv(dedup: dict, tsv_path: str):
    """Write directly to destination — avoids NFS/Lustre tmp-rename failures."""
    with open(tsv_path, "w", buffering=64 * 1024 * 1024) as f:
        f.write("key\tyear\n")
        for key, year in dedup.items():
            f.write(f"{key}\t{year}\n")


def merge_dedup_tsvs(worker_paths: list, merged_path: str) -> int:
    """
    Merge all per-worker dedup TSVs into one canonical TSV.
    For duplicate keys, keeps the highest year value.
    Safe to include TSVs from runs with different reader counts.
    """
    merged = {}
    for wp in worker_paths:
        if not os.path.exists(wp):
            continue
        with open(wp, "r", buffering=64 * 1024 * 1024) as f:
            next(f, None)
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                tab = line.rfind("\t")
                if tab == -1:
                    continue
                try:
                    key, year = line[:tab], int(line[tab + 1:])
                    if key not in merged or year > merged[key]:
                        merged[key] = year
                except ValueError:
                    continue
    save_dedup_tsv(merged, merged_path)
    return len(merged)


def slim(raw: dict) -> dict:
    return {k: v for k, v in raw.items() if k in KEEP_FIELDS}


# ─────────────────────────────────────────────────────────────────────────────
# COMPLETED-FILES MANIFEST
# Keyed by gz BASENAME (not worker index) so it survives gpu_config changes.
# ─────────────────────────────────────────────────────────────────────────────

def load_completed_files(manifest_path: str) -> set:
    """Return set of .gz basenames fully processed in a previous run."""
    completed = set()
    if not os.path.exists(manifest_path):
        return completed
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                completed.add(line)
    return completed


def mark_file_completed(manifest_path: str, gz_basename: str):
    """Append a fully-processed .gz basename to the shared manifest (append-safe)."""
    with open(manifest_path, "a") as f:
        f.write(gz_basename + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# PUBLICATION STATUS
# ─────────────────────────────────────────────────────────────────────────────

def is_published_or_accepted(raw: dict) -> bool:
    """
    Returns True if ANY location reports is_published or is_accepted.
    Records with no location data pass through (let BERT decide).
    """
    locations = raw.get("locations")
    if not locations:
        return True
    for loc in locations:
        if loc.get("is_published") or loc.get("is_accepted"):
            return True
    primary = raw.get("primary_location") or {}
    if primary.get("is_published") or primary.get("is_accepted"):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL FILTER
# ─────────────────────────────────────────────────────────────────────────────

def structural_filter(raw: dict, year_start: int, year_end: int) -> tuple:
    """Returns (passes: bool, rejection_reason: str). Ordered cheapest-first."""
    y = raw.get("publication_year")
    if isinstance(y, int) and not (year_start <= y <= year_end):
        return False, "year"
    if raw.get("is_retracted") or raw.get("is_paratext"):
        return False, "retracted"
    lang = (raw.get("language") or "").strip().lower()
    if lang and lang != "en":
        return False, "language"
    work_type = (raw.get("type") or "").strip().lower()
    if work_type and work_type not in ALLOWED_TYPES:
        return False, "type"
    if not is_published_or_accepted(raw):
        return False, "status"
    title = (raw.get("display_name") or raw.get("title") or "").strip()
    if not title:
        return False, "title"
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# READER WORKER
# ─────────────────────────────────────────────────────────────────────────────

def reader_worker(
    worker_id, file_list, in_queue, cfg, chunk_size,
    year_start, year_end,
    checked_val, kept_val, deduped_val,
    filtered_type_val, filtered_status_val,
    worker_tsv, initial_dedup,
    manifest_path,       # shared completed-files manifest (basename-keyed)
    skipped_files_val,   # shared int64: total .gz files skipped at file level
):
    try:
        # Private dedup copy per worker — merged back to dedup.tsv at pipeline end
        dedup = dict(initial_dedup)

        chunk_papers = []
        chunk_texts  = []

        local_checked  = 0
        local_kept     = 0
        local_deduped  = 0
        local_ftype    = 0
        local_fstatus  = 0

        records_since_dedup_sync = 0

        # Load completed-files manifest — keyed by basename, config-independent
        completed_basenames = load_completed_files(manifest_path)
        n_skipped = sum(1 for gz in file_list
                        if os.path.basename(gz) in completed_basenames)
        if n_skipped:
            with skipped_files_val.get_lock():
                skipped_files_val.value += n_skipped
            print(f"[reader-{worker_id}] Skipping {n_skipped}/{len(file_list)} "
                  f"already-completed files", flush=True)

        def flush_counters():
            nonlocal local_checked, local_kept, local_deduped, local_ftype, local_fstatus
            with checked_val.get_lock():         checked_val.value         += local_checked
            with kept_val.get_lock():            kept_val.value            += local_kept
            with deduped_val.get_lock():         deduped_val.value         += local_deduped
            with filtered_type_val.get_lock():   filtered_type_val.value   += local_ftype
            with filtered_status_val.get_lock(): filtered_status_val.value += local_fstatus
            local_checked = local_kept = local_deduped = local_ftype = local_fstatus = 0

        def try_save_dedup():
            try:
                save_dedup_tsv(dedup, worker_tsv)
            except Exception as e:
                print(f"[reader-{worker_id}] WARNING: dedup save failed (non-fatal): {e}",
                      flush=True)

        def flush_chunk():
            """Push current chunk to GPU queue. Blocks indefinitely — never drops records."""
            nonlocal chunk_papers, chunk_texts
            if not chunk_papers:
                return
            while True:
                try:
                    in_queue.put((chunk_papers, chunk_texts), timeout=30)
                    chunk_papers = []
                    chunk_texts  = []
                    return
                except Exception:
                    print(f"[reader-{worker_id}] Queue full, waiting for GPU...", flush=True)
                    time.sleep(5)

        for gz_path in file_list:
            gz_base = os.path.basename(gz_path)

            # File-level skip: basename-keyed, survives gpu_config changes
            if gz_base in completed_basenames:
                continue

            file_ok = True
            try:
                with subprocess.Popen(
                    ["pigz", "-dc", gz_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                ) as proc:
                    for raw_line in proc.stdout:
                        if not raw_line:
                            continue
                        try:
                            rec = orjson.loads(raw_line)
                        except Exception:
                            continue

                        local_checked            += 1
                        records_since_dedup_sync += 1

                        passes, reason = structural_filter(rec, year_start, year_end)
                        if not passes:
                            if reason == "type":     local_ftype   += 1
                            elif reason == "status": local_fstatus += 1
                            if local_checked % COUNTER_FLUSH_INTERVAL == 0:
                                flush_counters()
                            if records_since_dedup_sync >= DEDUP_SYNC_INTERVAL:
                                try_save_dedup()
                                records_since_dedup_sync = 0
                            continue

                        oaid = (rec.get("id")  or "").strip()
                        doi  = (rec.get("doi") or "").strip().lower()
                        year = rec.get("publication_year") or 0
                        key  = doi if doi else oaid

                        if key:
                            if year <= dedup.get(key, -1):
                                local_deduped += 1
                                if local_checked % COUNTER_FLUSH_INTERVAL == 0:
                                    flush_counters()
                                if records_since_dedup_sync >= DEDUP_SYNC_INTERVAL:
                                    try_save_dedup()
                                    records_since_dedup_sync = 0
                                continue
                            dedup[key] = year

                        text = (
                            build_text(rec, cfg)
                            or rec.get("display_name")
                            or rec.get("title")
                            or "unknown"
                        )
                        chunk_papers.append(slim(rec))
                        chunk_texts.append(text)
                        local_kept += 1

                        if local_checked % COUNTER_FLUSH_INTERVAL == 0:
                            flush_counters()
                        if records_since_dedup_sync >= DEDUP_SYNC_INTERVAL:
                            try_save_dedup()
                            records_since_dedup_sync = 0
                        if len(chunk_papers) >= chunk_size:
                            flush_chunk()   # blocks until GPU has space — no records lost

            except Exception as e:
                print(f"[reader-{worker_id}] ERROR on {gz_path}: {e}", flush=True)
                traceback.print_exc()
                file_ok = False

            # Only mark complete if the file was processed without exception
            if file_ok:
                mark_file_completed(manifest_path, gz_base)
                completed_basenames.add(gz_base)

        flush_chunk()
        flush_counters()

    except Exception as e:
        print(f"[reader-{worker_id}] FATAL: {e}", flush=True)
        traceback.print_exc()
    finally:
        try:
            save_dedup_tsv(dedup, worker_tsv)
        except Exception:
            pass
        in_queue.put(None)


# ─────────────────────────────────────────────────────────────────────────────
# GPU INFERENCE WORKER
# ─────────────────────────────────────────────────────────────────────────────

def gpu_worker(in_queue, out_queue, model_path, batch_size, device,
               max_len, num_readers, threshold):
    assert batch_size > 0,          "Batch size must be a positive integer."
    assert 0.0 <= threshold <= 1.0, "Threshold must be between 0.0 and 1.0."

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model     = (AutoModelForSequenceClassification
                     .from_pretrained(model_path)
                     .to(device)
                     .eval())

        def infer_gpu(papers: list, texts: list) -> np.ndarray:
            n         = len(papers)
            probs_cpu = np.zeros(n, dtype=np.float32)
            amp_ctx   = (torch.autocast("cuda", dtype=torch.bfloat16)
                         if "cuda" in device else nullcontext())

            with torch.inference_mode(), amp_ctx:
                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    try:
                        enc = tokenizer(
                            texts[start:end],
                            padding=True, truncation=True,
                            max_length=max_len, return_tensors="pt",
                        )
                        inputs = {
                            "input_ids":      enc["input_ids"].to(device, non_blocking=True),
                            "attention_mask": enc["attention_mask"].to(device, non_blocking=True),
                        }
                        if "token_type_ids" in enc:
                            inputs["token_type_ids"] = enc["token_type_ids"].to(
                                device, non_blocking=True)

                        logits = model(**inputs).logits.squeeze(-1)
                        probs_cpu[start:end] = torch.sigmoid(logits).float().cpu().numpy()
                        del inputs, logits, enc

                        if start % (batch_size * 4) == 0:
                            torch.cuda.empty_cache()

                    except torch.cuda.OutOfMemoryError:
                        print(f"[gpu:{device}] OOM at batch {start}-{end}, "
                              f"halving to micro-batches...", flush=True)
                        torch.cuda.empty_cache()
                        micro = max(batch_size // 4, 1)
                        for ss in range(start, end, micro):
                            se  = min(ss + micro, end)
                            enc = tokenizer(
                                texts[ss:se], padding=True, truncation=True,
                                max_length=max_len, return_tensors="pt",
                            )
                            inp    = {k: v.to(device) for k, v in enc.items()}
                            logits = model(**inp).logits.squeeze(-1)
                            probs_cpu[ss:se] = torch.sigmoid(logits).float().cpu().numpy()
                            del inp, logits, enc
                        torch.cuda.empty_cache()

            return probs_cpu

        done_count = 0
        while done_count < num_readers:
            item = in_queue.get()
            if item is None:
                done_count += 1
                continue
            papers, texts = item
            probs     = infer_gpu(papers, texts)
            positives = [
                (paper, round(float(prob), 4))
                for paper, prob in zip(papers, probs.tolist())
                if prob >= threshold
            ]
            if positives:
                out_queue.put(positives)
            del papers, texts, probs, positives

        out_queue.put(None)   # signal writer that this GPU is done

    except Exception as e:
        print(f"\n[gpu:{device}] FATAL ERROR: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        out_queue.put(None)   # always send sentinel so writer can exit cleanly
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# WRITER WORKER
# ─────────────────────────────────────────────────────────────────────────────

def writer_worker(out_queue, out_path, threshold, pos_val, num_gpu_workers):
    """
    Receives results from all GPU workers via out_queue.
    Expects exactly `num_gpu_workers` None sentinels before exiting.
    Has a per-get timeout so it never hangs if a GPU worker is hard-killed.
    """
    assert threshold >= 0.0, "Threshold must be explicitly provided."
    try:
        local_pos     = 0
        unflushed_pos = 0
        SYNC_INTERVAL = 200
        done_count    = 0

        with open(out_path, "a", encoding="utf-8", buffering=8 * 1024 * 1024) as f:
            while done_count < num_gpu_workers:
                try:
                    item = out_queue.get(timeout=WRITER_QUEUE_TIMEOUT_S)
                except Exception:
                    # Timeout — GPU worker(s) likely hard-killed
                    print(
                        f"[writer] {WRITER_QUEUE_TIMEOUT_S}s timeout waiting for GPU output "
                        f"({done_count}/{num_gpu_workers} sentinels received). "
                        f"GPU workers may have crashed. Flushing and exiting.",
                        flush=True,
                    )
                    break

                if item is None:
                    done_count += 1
                    continue

                for paper, prob in item:
                    local_pos     += 1
                    unflushed_pos += 1
                    out = dict(paper)
                    out.update({
                        "bert_score":     prob,
                        "bert_label":     1,
                        "bert_threshold": threshold,
                    })
                    f.write(orjson.dumps(out).decode("utf-8") + "\n")

                if local_pos > 0:
                    with pos_val.get_lock():
                        pos_val.value += local_pos
                    local_pos = 0

                if unflushed_pos >= SYNC_INTERVAL:
                    f.flush()
                    os.fsync(f.fileno())
                    unflushed_pos = 0

            # Final flush regardless of how we exited the loop
            if unflushed_pos > 0:
                f.flush()
                os.fsync(f.fileno())

            if local_pos > 0:
                with pos_val.get_lock():
                    pos_val.value += local_pos

    except Exception as e:
        print(f"\n[writer] FATAL ERROR: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# SEED-STABLE FILE PARTITIONING
# ─────────────────────────────────────────────────────────────────────────────

def partition_files(gz_files: list, gpu_configs: List[GpuConfig], seed: int) -> Dict:
    """
    Assign .gz files to (gpu_idx, reader_idx) slots deterministically.

    Guarantees:
      - Same seed + same file list = identical assignment on every restart
      - No file assigned to more than one slot (strictly non-overlapping)
      - Changing gpu_configs reshuffles assignments but remains non-overlapping

    Returns dict: (gpu_idx, reader_idx) -> list of gz paths
    """
    import random
    rng   = random.Random(seed)
    files = list(gz_files)
    rng.shuffle(files)

    slots = []
    for gi, gcfg in enumerate(gpu_configs):
        for ri in range(gcfg.num_readers):
            slots.append((gi, ri))

    partitions: Dict = {slot: [] for slot in slots}
    for i, fpath in enumerate(files):
        partitions[slots[i % len(slots)]].append(fpath)

    return partitions


# ─────────────────────────────────────────────────────────────────────────────
# BACKUP UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def do_backup_and_merge(output_dir: str):
    """
    1. Auto-detect all dedup_worker_*.tsv files in output_dir
    2. Merge them into dedup.tsv
    3. Backup dedup.tsv and pass1_bert.jsonl with timestamp suffixes

    Safe to run while the pipeline is live (reads TSVs; does not modify them).
    Run this BEFORE killing a live pipeline to capture full dedup state.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    worker_paths = sorted(glob(os.path.join(output_dir, "dedup_worker_*.tsv")))
    merged_tsv   = os.path.join(output_dir, "dedup.tsv")

    if worker_paths:
        print(f"[backup] Found {len(worker_paths)} worker TSVs, merging → {merged_tsv}")
        n = merge_dedup_tsvs(worker_paths, merged_tsv)
        print(f"[backup] Merged {n:,} unique dedup keys.")
    else:
        print("[backup] No worker TSVs found — skipping merge.")

    for fname in ["dedup.tsv", "pass1_bert.jsonl"]:
        src = os.path.join(output_dir, fname)
        dst = os.path.join(output_dir, f"{fname}.backup.{ts}")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            size_mb = os.path.getsize(dst) / 1e6
            print(f"[backup] {fname}  →  {os.path.basename(dst)}  ({size_mb:.1f} MB)")
        else:
            print(f"[backup] WARNING: {src} not found — skipping backup.")

    print("[backup] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--snapshot_dir", default=None)
    ap.add_argument("--output_dir",   required=True)
    ap.add_argument("--model",        default=None)
    ap.add_argument("--config",       default="bert_classifier_config.yaml")
    ap.add_argument("--threshold",    type=float, default=None)
    ap.add_argument("--year_start",   type=int,   default=2000)
    ap.add_argument("--year_end",     type=int,   default=2026)
    ap.add_argument("--pattern",      default="**/*.gz")
    ap.add_argument("--seed",         type=int,   default=42,
                    help="RNG seed for file->worker assignment. "
                         "Keep constant across runs for reproducible partitioning.")
    ap.add_argument("--resume",       action="store_true",
                    help="Resume: load dedup.tsv, skip completed files, "
                         "auto-backup pass1_bert.jsonl before appending.")

    # Per-GPU config — repeat once per GPU.
    # Format: "DEVICE_ID:batch=N,chunk=N,queue=N,readers=N"
    ap.add_argument(
        "--gpu_configs", nargs="+",
        default=["0:batch=2048,chunk=16384,queue=4,readers=2"],
        metavar="GPU_SPEC",
        help=(
            "Per-GPU config string(s). Safe to change between runs — "
            "resume state is preserved via basename-keyed manifest. "
            "Example: \"0:batch=1024,chunk=16384,queue=3,readers=2\" "
            "         \"1:batch=2048,chunk=16384,queue=4,readers=3\""
        ),
    )

    # Utility modes (no pipeline run)
    ap.add_argument("--backup-only", action="store_true",
                    help="Merge worker TSVs + backup outputs, then exit. "
                         "Safe to run against a live pipeline.")
    ap.add_argument("--merge-only",  action="store_true",
                    help="Merge worker TSVs into dedup.tsv only, then exit.")

    args = ap.parse_args()

    # ── Utility modes ─────────────────────────────────────────────────────────
    if args.backup_only:
        do_backup_and_merge(args.output_dir)
        return
    if args.merge_only:
        worker_paths = sorted(glob(os.path.join(args.output_dir, "dedup_worker_*.tsv")))
        merged_tsv   = os.path.join(args.output_dir, "dedup.tsv")
        print(f"[merge] Found {len(worker_paths)} worker TSVs")
        n = merge_dedup_tsvs(worker_paths, merged_tsv)
        print(f"[merge] Wrote {n:,} keys -> {merged_tsv}")
        return

    # ── Validate pipeline args ────────────────────────────────────────────────
    if not args.snapshot_dir:
        ap.error("--snapshot_dir is required for pipeline runs.")
    if not args.model:
        ap.error("--model is required for pipeline runs.")

    # ── Parse and validate GPU configs ───────────────────────────────────────
    gpu_configs: List[GpuConfig] = [parse_gpu_config(s) for s in args.gpu_configs]
    total_readers = sum(g.num_readers for g in gpu_configs)

    if torch.cuda.is_available():
        n_avail = torch.cuda.device_count()
        print(f"\n[gpu] {n_avail} CUDA device(s) visible:")
        for gc in gpu_configs:
            if gc.device_id >= n_avail:
                raise ValueError(
                    f"Requested cuda:{gc.device_id} but only {n_avail} GPU(s) visible. "
                    f"Check CUDA_VISIBLE_DEVICES."
                )
            free_mem, total_mem = torch.cuda.mem_get_info(gc.device_id)
            print(f"  {gc}  |  free={free_mem/1e9:.1f}/{total_mem/1e9:.1f} GB")
    else:
        print("  WARNING: No CUDA GPUs found — running on CPU (very slow).")
        gpu_configs = [GpuConfig(device_id=-1, batch_size=64, chunk_size=512,
                                 queue_max=2, num_readers=1)]

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    threshold = (args.threshold if args.threshold is not None
                 else float(cfg["inference"]["threshold"]))
    max_len   = int(cfg["model"]["max_len"])

    out_jsonl     = os.path.join(args.output_dir, "pass1_bert.jsonl")
    dedup_tsv     = os.path.join(args.output_dir, "dedup.tsv")
    manifest_path = os.path.join(args.output_dir, COMPLETED_MANIFEST_NAME)

    # Worker dedup TSVs — one per reader, globally indexed
    worker_tsvs = [
        os.path.join(args.output_dir, f"dedup_worker_{i}.tsv")
        for i in range(total_readers)
    ]

    gz_files = sorted(
        glob(os.path.join(args.snapshot_dir, args.pattern), recursive=True),
        key=os.path.getsize, reverse=True,
    )
    gz_files = [f for f in gz_files if f.endswith(".gz")]
    if not gz_files:
        raise FileNotFoundError(f"No .gz files found under {args.snapshot_dir}.")

    # ── Resume or clean start ─────────────────────────────────────────────────
    if not args.resume:
        for path in [out_jsonl, dedup_tsv, manifest_path] + worker_tsvs:
            if os.path.exists(path):
                os.remove(path)
        initial_dedup = {}
        print("[clean start] All prior state cleared.")
    else:
        initial_dedup = load_dedup_tsv(dedup_tsv)
        print(f"[resume] Loaded {len(initial_dedup):,} dedup entries from {dedup_tsv}")

        # Auto-backup pass1_bert.jsonl before appending (separate resume backup)
        if os.path.exists(out_jsonl):
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            bak = f"{out_jsonl}.resume_backup.{ts}"
            shutil.copy2(out_jsonl, bak)
            size_mb = os.path.getsize(bak) / 1e6
            print(f"[resume] Backed up pass1_bert.jsonl -> "
                  f"{os.path.basename(bak)}  ({size_mb:.1f} MB)")

        # Report how many files will be skipped at file level
        completed  = load_completed_files(manifest_path)
        total_gz   = len(gz_files)
        skippable  = sum(1 for f in gz_files if os.path.basename(f) in completed)
        remaining  = total_gz - skippable
        print(f"[resume] {skippable}/{total_gz} .gz files already completed "
              f"({remaining} remaining).")
        if remaining == 0:
            print("[resume] Nothing to do — all files already processed. Exiting cleanly.")
            sys.exit(0)

    # ── Seed-stable partitioning ──────────────────────────────────────────────
    partitions = partition_files(gz_files, gpu_configs, seed=args.seed)

    # ── Shared counters ───────────────────────────────────────────────────────
    checked_val         = Value(ctypes.c_int64, 0)
    kept_val            = Value(ctypes.c_int64, 0)
    deduped_val         = Value(ctypes.c_int64, 0)
    pos_val             = Value(ctypes.c_int64, 0)
    filtered_type_val   = Value(ctypes.c_int64, 0)
    filtered_status_val = Value(ctypes.c_int64, 0)
    skipped_files_val   = Value(ctypes.c_int64, 0)

    # ── Build queues, GPU workers, readers ───────────────────────────────────
    out_queue        = Queue(maxsize=8)
    in_queues: List  = []
    gpu_procs: List  = []
    readers:   List  = []
    global_reader_id = 0

    for gi, gcfg in enumerate(gpu_configs):
        dev = gcfg.device if torch.cuda.is_available() else "cpu"
        iq  = Queue(maxsize=max(gcfg.queue_max, 1))
        in_queues.append(iq)

        gp = Process(
            target=gpu_worker,
            args=(iq, out_queue, args.model, gcfg.batch_size,
                  dev, max_len, gcfg.num_readers, threshold),
            daemon=True,
        )
        gp.start()
        gpu_procs.append(gp)
        print(f"[main] GPU worker started on {dev} "
              f"(batch={gcfg.batch_size}, readers={gcfg.num_readers})", flush=True)

        for ri in range(gcfg.num_readers):
            wid       = global_reader_id
            file_list = partitions.get((gi, ri), [])
            p = Process(
                target=reader_worker,
                args=(
                    wid, file_list, iq, cfg, gcfg.chunk_size,
                    args.year_start, args.year_end,
                    checked_val, kept_val, deduped_val,
                    filtered_type_val, filtered_status_val,
                    worker_tsvs[wid], initial_dedup,
                    manifest_path,
                    skipped_files_val,
                ),
                daemon=True,
            )
            p.start()
            readers.append(p)
            print(f"[main] reader-{wid} -> {dev}  "
                  f"({len(file_list)} files, chunk={gcfg.chunk_size})", flush=True)
            global_reader_id += 1

    del initial_dedup   # free parent-process memory

    writer_proc = Process(
        target=writer_worker,
        args=(out_queue, out_jsonl, threshold, pos_val, len(gpu_configs)),
        daemon=True,
    )
    writer_proc.start()

    # ── Signal handlers ───────────────────────────────────────────────────────
    def kill_all():
        try:
            for iq in in_queues:
                while not iq.empty():
                    iq.get_nowait()
            while not out_queue.empty():
                out_queue.get_nowait()
        except Exception:
            pass
        for iq in in_queues:
            iq.cancel_join_thread()
        out_queue.cancel_join_thread()
        for p in readers + gpu_procs + [writer_proc]:
            if p.is_alive():
                p.terminate()
        time.sleep(1)
        for p in readers + gpu_procs + [writer_proc]:
            if p.is_alive():
                p.kill()

    def _shutdown(sig, frame):
        print(f"\n[signal {sig}] Shutting down cleanly...", flush=True)
        kill_all()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    # ── Startup summary ───────────────────────────────────────────────────────
    print(f"\n[{datetime.now():%H:%M:%S}] Pipeline started")
    print(f"  Total .gz files : {len(gz_files):,}")
    print(f"  Total readers   : {total_readers}")
    print(f"  GPU workers     : {len(gpu_configs)}")
    print(f"  BERT threshold  : {threshold}")
    print(f"  Year range      : {args.year_start}-{args.year_end}")
    print(f"  Allowed types   : {', '.join(sorted(ALLOWED_TYPES))}")
    print(f"  Dedup sync      : every {DEDUP_SYNC_INTERVAL:,} records/worker")
    print(f"  Partition seed  : {args.seed}  (keep constant for reproducible resume)")
    print(f"  Resume mode     : {args.resume}")
    for gi, gc in enumerate(gpu_configs):
        dev = gc.device if torch.cuda.is_available() else "cpu"
        print(f"    GPU {gi}: {dev}  batch={gc.batch_size}  chunk={gc.chunk_size}  "
              f"queue={gc.queue_max}  readers={gc.num_readers}")
    print()

    # ── Monitoring loop ───────────────────────────────────────────────────────
    t0, last_checked = time.time(), 0
    pbar = tqdm(
        desc="Scanning",
        unit="rec",
        dynamic_ncols=True,
        mininterval=2.0,
        bar_format="{l_bar}{bar}| {n_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    while True:
        time.sleep(5)

        checked  = checked_val.value
        kept     = kept_val.value
        deduped  = deduped_val.value
        pos      = pos_val.value
        ftype    = filtered_type_val.value
        fstatus  = filtered_status_val.value
        skipped  = skipped_files_val.value
        elapsed  = max(time.time() - t0, 1)
        rate     = checked / elapsed

        pbar.update(checked - last_checked)
        last_checked = checked
        pass_rate = 100.0 * kept / max(checked, 1)
        iq_sizes  = "+".join(str(iq.qsize()) for iq in in_queues)

        pbar.set_postfix({
            "checked":  f"{checked/1e6:.2f}M",
            "pass%":    f"{pass_rate:.1f}%",
            "Δtype":    f"{ftype/1e3:.1f}k",
            "Δstatus":  f"{fstatus/1e3:.1f}k",
            "deduped":  f"{deduped/1e3:.1f}k",
            "BERT+":    f"{pos:,}",
            "rec/s":    f"{rate:,.0f}",
            "iq":       iq_sizes,
            "skip_f":   skipped,
        }, refresh=True)

        gpu_alive     = any(p.is_alive() for p in gpu_procs)
        writer_alive  = writer_proc.is_alive()
        readers_alive = any(p.is_alive() for p in readers)

        if not readers_alive and not gpu_alive and not writer_alive:
            break
        if not gpu_alive and readers_alive:
            print("\n[FATAL] All GPU workers crashed. Shutting down.", flush=True)
            kill_all()
            sys.exit(1)
        if not writer_alive and (gpu_alive or readers_alive):
            print("\n[FATAL] Writer worker crashed. Shutting down.", flush=True)
            kill_all()
            sys.exit(1)

    pbar.close()
    for p in readers:   p.join(timeout=60)
    for p in gpu_procs: p.join(timeout=60)
    writer_proc.join(timeout=60)

    # ── Final dedup merge ─────────────────────────────────────────────────────
    print(f"\n[merge] Merging worker TSVs -> {dedup_tsv} ...")
    all_worker_tsvs = sorted(glob(os.path.join(args.output_dir, "dedup_worker_*.tsv")))
    n_merged = merge_dedup_tsvs(all_worker_tsvs, dedup_tsv)
    print(f"[merge] {n_merged:,} unique keys written.")
    for wp in worker_tsvs:
        if os.path.exists(wp):
            os.remove(wp)

    # ── Final summary ─────────────────────────────────────────────────────────
    dt      = time.time() - t0
    checked = checked_val.value
    kept    = kept_val.value
    pos     = pos_val.value
    ftype   = filtered_type_val.value
    fstatus = filtered_status_val.value
    deduped = deduped_val.value
    skipped = skipped_files_val.value
    other   = max(checked - kept - ftype - fstatus - deduped, 0)

    print("=" * 65)
    print(f"[{datetime.now():%H:%M:%S}] PIPELINE COMPLETE")
    print(f"  Files skipped (resume)    : {skipped:>12,}")
    print(f"  Records checked           : {checked:>12,}")
    print(f"  |-  year/lang/retracted   : {other:>12,}")
    print(f"  |-  filtered by type      : {ftype:>12,}")
    print(f"  |-  filtered by status    : {fstatus:>12,}")
    print(f"  |-  deduplicated          : {deduped:>12,}")
    print(f"  +-  sent to BERT          : {kept:>12,}  ({100*kept/max(checked,1):.2f}%)")
    print(f"  BERT positives            : {pos:>12,}  (threshold={threshold})")
    print(f"  Time                      : {dt/60:>11.1f} min")
    print(f"  Throughput                : {checked/max(dt,1):>10,.0f} rec/s")
    print("=" * 65)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
