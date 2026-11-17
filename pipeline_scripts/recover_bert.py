#!/usr/bin/env python3
"""
recover_bert_pass.py
====================
Recovery script: re-run BERT inference on records that passed structural
filtering and dedup in the main pipeline but were never written to
pass1_bert.jsonl (e.g. due to a GPU/writer crash mid-run).

THREE-PHASE DESIGN:
═══════════════════

  Phase 0 — Load state  (main process)
    1. Load pass1_bert.jsonl  → "already_have" set  (keys already written)
    2. Load dedup.tsv         → "candidate_keys" set (structural+dedup survivors)

  Phase 1 — Parallel pre-scan  (N CPU workers, no GPU)
    - Same multiprocess architecture as main pipeline readers
    - Uses pigz + orjson + parallel file partitioning
    - Each worker scans its file partition through all 5 gates
    - Workers report which files contain missing candidates + exact counts
    - Output: recovery_files.txt listing ONLY .gz files with missing candidates
    - Phase 2 re-scans only those files (memory + time efficient)
    - SKIP with --skip_prescan to go straight to Phase 2 on ALL files.
      Use this when missing candidates are spread across the full dataset
      (i.e. dedup sync happened across many files before the crash).

  Phase 2 — Recovery inference  (multi-GPU, identical to main pipeline)
    - Re-scans files (recovery_files.txt or all files if --skip_prescan)
    - Same 5-gate logic as prescan
    - Sends survivors to BERT → appends positives to pass1_bert.jsonl

5 GATES (cheapest first — applied identically in both prescan and Phase 2):
  Gate 1: structural_filter        (year/lang/type/status/title) — ~88% rejected
  Gate 2: key exists               (doi preferred, else oaid)
  Gate 3: key in candidate_keys    (was a real BERT candidate per dedup.tsv)
  Gate 4: key NOT in already_have  (not already written to pass1_bert.jsonl)
  Gate 5: key NOT in local_seen    (within-worker dedup)

THROUGHPUT (--skip_prescan on 480M records):
  Gate 1 rejects ~422M immediately (cheapest CPU check)
  Gate 3 rejects most remaining via set lookup
  Net ~50M records reach BERT — much faster than a clean pipeline run.

RUN — skip prescan, two GPUs (recommended when crash spread across many files):
    CUDA_VISIBLE_DEVICES=0,1 nohup python -u recover_bert_pass.py \\
      --snapshot_dir /path/to/openalex-works \\
      --output_dir   /path/to/corpus_output \\
      --model        /path/to/hf_model \\
      --config       bert_classifier_config.yaml \\
      --gpu_configs  "0:batch=1024,chunk=16384,queue=4,readers=2" \\
                     "1:batch=1024,chunk=16384,queue=4,readers=2" \\
      --threshold    0.45 --year_start 2000 --year_end 2026 --seed 42 \\
      --skip_prescan \\
      > logs/recover_bert_pass.log 2>&1 &

RUN — with prescan (use when you expect only a few files affected):
    python -u recover_bert_pass.py ... (same args, without --skip_prescan)

PRE-SCAN ONLY (no GPU, no output written):
    python -u recover_bert_pass.py \\
      --snapshot_dir ... --output_dir ... --config ... \\
      --prescan-only --prescan_workers 8

NOTES:
    - pass1_bert.jsonl is auto-backed up before any writes.
    - Recovered papers tagged "recovery_pass": true in JSONL.
    - WRITER_SYNC_INTERVAL = 500 (vs 10,000 in main pipeline) for
      tighter crash safety — we never want to re-run this again.
    - Run --backup-only on the main pipeline FIRST to ensure dedup.tsv
      is fully merged before running this script.
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
# CONSTANTS — identical to main pipeline
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
    "article", "preprint", "book", "book-chapter", "dataset",
    "dissertation", "review", "proceedings-article", "report",
    "standard", "other",
})

COUNTER_FLUSH_INTERVAL = 1_000
WRITER_QUEUE_TIMEOUT_S = 300   # writer exits if no GPU output for 5 min
WRITER_SYNC_INTERVAL   = 500   # fsync every N positives (tight crash safety)
RECOVERY_FILES_NAME    = "recovery_files.txt"


# ─────────────────────────────────────────────────────────────────────────────
# PER-GPU CONFIG — identical to main pipeline
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GpuConfig:
    """Configuration for one GPU worker and its dedicated reader pool."""
    device_id:   int = 0
    batch_size:  int = 1024
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
    """Parse "DEVICE_ID:batch=N,chunk=N,queue=N,readers=N" into GpuConfig."""
    if ":" not in spec:
        return GpuConfig(device_id=int(spec.strip()))
    dev_str, rest = spec.split(":", 1)
    cfg = GpuConfig(device_id=int(dev_str.strip()))
    for kv in rest.split(","):
        kv = kv.strip()
        if not kv:
            continue
        if "=" not in kv:
            raise ValueError(f"Malformed key=value in GPU spec: {kv!r}")
        k, v = kv.split("=", 1)
        k, v = k.strip(), int(v.strip())
        if   k == "batch":   cfg.batch_size  = v
        elif k == "chunk":   cfg.chunk_size  = v
        elif k == "queue":   cfg.queue_max   = v
        elif k == "readers": cfg.num_readers = v
        else:
            raise ValueError(f"Unknown GPU config key: {k!r}")
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL FILTER — identical to main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def is_published_or_accepted(raw: dict) -> bool:
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


def structural_filter(raw: dict, year_start: int, year_end: int) -> bool:
    """True if record passes all structural gates. Cheapest checks first."""
    y = raw.get("publication_year")
    if isinstance(y, int) and not (year_start <= y <= year_end):
        return False
    if raw.get("is_retracted") or raw.get("is_paratext"):
        return False
    lang = (raw.get("language") or "").strip().lower()
    if lang and lang != "en":
        return False
    work_type = (raw.get("type") or "").strip().lower()
    if work_type and work_type not in ALLOWED_TYPES:
        return False
    if not is_published_or_accepted(raw):
        return False
    title = (raw.get("display_name") or raw.get("title") or "").strip()
    if not title:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# KEY + SLIM HELPERS — identical to main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def make_key(rec: dict) -> str:
    """Dedup key: doi preferred, else oaid. Matches main pipeline exactly."""
    doi  = (rec.get("doi") or "").strip().lower()
    oaid = (rec.get("id")  or "").strip()
    return doi if doi else oaid


def slim(raw: dict) -> dict:
    return {k: v for k, v in raw.items() if k in KEEP_FIELDS}


# ─────────────────────────────────────────────────────────────────────────────
# STATE LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_candidate_keys(tsv_path: str) -> set:
    """
    Load dedup.tsv → SET of keys that survived structural_filter + dedup.
    These are the records that were (or should have been) sent to BERT.
    RAM: ~5-8 GB for 100M keys as a Python set.
    See bloom filter note at bottom of file if OOM.
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(
            f"dedup.tsv not found: {tsv_path}\n"
            f"Run the main pipeline with --backup-only first to merge worker TSVs."
        )
    keys = set()
    print(f"[load] Reading candidate keys from {tsv_path} ...", flush=True)
    t0 = time.time()
    with open(tsv_path, "r", buffering=64 * 1024 * 1024) as f:
        next(f, None)  # skip header "key\tyear"
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            tab = line.rfind("\t")
            if tab == -1:
                continue
            keys.add(line[:tab])
    print(f"[load] {len(keys):,} candidate keys loaded in {time.time()-t0:.1f}s",
          flush=True)
    return keys


def load_already_have(jsonl_path: str) -> set:
    """
    Load pass1_bert.jsonl → SET of doi/oaid keys already written.
    These are skipped in recovery to avoid duplicates.
    """
    have = set()
    if not os.path.exists(jsonl_path):
        print(f"[load] {jsonl_path} not found — 0 papers already written.",
              flush=True)
        return have
    print(f"[load] Building already_have set from {jsonl_path} ...", flush=True)
    t0 = time.time()
    with open(jsonl_path, "r", buffering=64 * 1024 * 1024) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = orjson.loads(line)
                key = make_key(rec)
                if key:
                    have.add(key)
            except Exception:
                continue
    print(f"[load] {len(have):,} already-written keys loaded in {time.time()-t0:.1f}s",
          flush=True)
    return have


# ─────────────────────────────────────────────────────────────────────────────
# FILE PARTITIONING
# ─────────────────────────────────────────────────────────────────────────────

def partition_files_simple(gz_files: list, n_workers: int, seed: int) -> List[list]:
    """Round-robin partition for prescan workers (not GPU-aware)."""
    import random
    rng   = random.Random(seed)
    files = list(gz_files)
    rng.shuffle(files)
    parts: List[list] = [[] for _ in range(n_workers)]
    for i, f in enumerate(files):
        parts[i % n_workers].append(f)
    return parts


def partition_files_gpu(gz_files: list, gpu_configs: List[GpuConfig],
                        seed: int) -> Dict:
    """
    Assign files to (gpu_idx, reader_idx) slots — identical to main pipeline.
    Same seed + same file list = identical assignment every restart.
    Files are strictly non-overlapping across all slots.
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
# PHASE 1: PARALLEL PRESCAN WORKER
# Each worker scans its file partition using pigz + orjson, applies 5 gates,
# and reports (gz_path, n_new) for every file that has missing candidates.
# ─────────────────────────────────────────────────────────────────────────────

def prescan_worker(
    worker_id:      int,
    file_list:      list,
    candidate_keys: set,     # read-only copy via spawn
    already_have:   set,     # read-only copy via spawn
    year_start:     int,
    year_end:       int,
    result_queue:   Queue,   # puts (gz_path, n_new) per file; None sentinel when done
    ps_checked:     Value,
    ps_struct:      Value,
    ps_in_dedup:    Value,
    ps_skip_have:   Value,
    ps_new:         Value,
    ps_files_done:  Value,
):
    try:
        local_seen   = set()
        lc_checked   = 0
        lc_struct    = 0
        lc_in_dedup  = 0
        lc_skip_have = 0
        lc_new       = 0

        def flush_counters():
            nonlocal lc_checked, lc_struct, lc_in_dedup, lc_skip_have, lc_new
            with ps_checked.get_lock():   ps_checked.value   += lc_checked
            with ps_struct.get_lock():    ps_struct.value    += lc_struct
            with ps_in_dedup.get_lock():  ps_in_dedup.value  += lc_in_dedup
            with ps_skip_have.get_lock(): ps_skip_have.value += lc_skip_have
            with ps_new.get_lock():       ps_new.value       += lc_new
            lc_checked   = 0
            lc_struct    = 0
            lc_in_dedup  = 0
            lc_skip_have = 0
            lc_new       = 0

        for gz_path in file_list:
            file_new = 0
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

                        lc_checked += 1

                        # Gate 1
                        if not structural_filter(rec, year_start, year_end):
                            if lc_checked % COUNTER_FLUSH_INTERVAL == 0:
                                flush_counters()
                            continue
                        lc_struct += 1

                        # Gate 2 + 3
                        key = make_key(rec)
                        if not key or key not in candidate_keys:
                            if lc_checked % COUNTER_FLUSH_INTERVAL == 0:
                                flush_counters()
                            continue
                        lc_in_dedup += 1

                        # Gate 4
                        if key in already_have:
                            lc_skip_have += 1
                            if lc_checked % COUNTER_FLUSH_INTERVAL == 0:
                                flush_counters()
                            continue

                        # Gate 5
                        if key in local_seen:
                            if lc_checked % COUNTER_FLUSH_INTERVAL == 0:
                                flush_counters()
                            continue
                        local_seen.add(key)

                        lc_new   += 1
                        file_new += 1

                        if lc_checked % COUNTER_FLUSH_INTERVAL == 0:
                            flush_counters()

            except Exception as e:
                print(f"[prescan-{worker_id}] WARNING: error on {gz_path}: {e}",
                      flush=True)

            # Report file to main process if it has any missing candidates
            if file_new > 0:
                result_queue.put((gz_path, file_new))

            with ps_files_done.get_lock():
                ps_files_done.value += 1

        flush_counters()

    except Exception as e:
        print(f"[prescan-{worker_id}] FATAL: {e}", flush=True)
        traceback.print_exc()
    finally:
        result_queue.put(None)  # sentinel — always sent


def prescan_parallel(
    gz_files:       list,
    candidate_keys: set,
    already_have:   set,
    year_start:     int,
    year_end:       int,
    n_workers:      int,
    seed:           int,
    output_dir:     str,
) -> tuple:
    """
    Run parallel prescan across n_workers processes.
    Returns (recovery_files: list, total_new_candidates: int).
    Writes recovery_files.txt to output_dir.
    """
    print(f"[prescan] Scanning {len(gz_files):,} .gz files using "
          f"{n_workers} parallel workers ...", flush=True)
    print(f"[prescan] (CPU-only pass — pigz + orjson, no GPU)\n", flush=True)

    parts = partition_files_simple(gz_files, n_workers, seed)

    ps_checked    = Value(ctypes.c_int64, 0)
    ps_struct     = Value(ctypes.c_int64, 0)
    ps_in_dedup   = Value(ctypes.c_int64, 0)
    ps_skip_have  = Value(ctypes.c_int64, 0)
    ps_new        = Value(ctypes.c_int64, 0)
    ps_files_done = Value(ctypes.c_int64, 0)

    result_queue = Queue()

    workers = []
    for wid, file_list in enumerate(parts):
        p = Process(
            target=prescan_worker,
            args=(
                wid, file_list,
                candidate_keys, already_have,
                year_start, year_end,
                result_queue,
                ps_checked, ps_struct, ps_in_dedup,
                ps_skip_have, ps_new, ps_files_done,
            ),
            daemon=True,
        )
        p.start()
        workers.append(p)
        print(f"[prescan] Worker {wid} started ({len(file_list)} files)", flush=True)

    t0           = time.time()
    recovery_map = {}   # gz_path -> total n_new across all workers
    sentinels    = 0
    last_done    = 0

    pbar = tqdm(
        total=len(gz_files),
        desc="prescan",
        unit="file",
        dynamic_ncols=True,
        mininterval=2.0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    while sentinels < n_workers:
        # Drain result queue
        while True:
            try:
                item = result_queue.get(timeout=1.0)
                if item is None:
                    sentinels += 1
                else:
                    gz_path, n_new = item
                    recovery_map[gz_path] = recovery_map.get(gz_path, 0) + n_new
            except Exception:
                break  # queue empty

        # Update tqdm
        done  = ps_files_done.value
        delta = done - last_done
        if delta > 0:
            pbar.update(delta)
        last_done = done

        checked  = ps_checked.value
        in_dedup = ps_in_dedup.value
        new      = ps_new.value
        elapsed  = max(time.time() - t0, 1)

        pbar.set_postfix({
            "checked":   f"{checked/1e6:.2f}M",
            "in_dedup":  f"{in_dedup/1e6:.2f}M",
            "new_cands": f"{new/1e3:.1f}k",
            "rec/s":     f"{checked/elapsed:,.0f}",
            "hit_files": f"{len(recovery_map)}",
        }, refresh=True)

    # Drain any remaining results after all sentinels received
    while True:
        try:
            item = result_queue.get_nowait()
            if item is not None:
                gz_path, n_new = item
                recovery_map[gz_path] = recovery_map.get(gz_path, 0) + n_new
        except Exception:
            break

    pbar.update(len(gz_files) - last_done)
    pbar.close()
    for p in workers:
        p.join(timeout=30)

    dt = time.time() - t0

    total_checked   = ps_checked.value
    total_struct    = ps_struct.value
    total_in_dedup  = ps_in_dedup.value
    total_skip_have = ps_skip_have.value
    total_new       = ps_new.value

    # Sort recovery files largest-first for faster Phase 2 startup
    recovery_files = sorted(
        recovery_map.keys(),
        key=os.path.getsize, reverse=True,
    )

    # Write manifest
    recovery_manifest = os.path.join(output_dir, RECOVERY_FILES_NAME)
    with open(recovery_manifest, "w") as f:
        for gz_path in recovery_files:
            f.write(gz_path + "\n")
    print(f"[prescan] Wrote {len(recovery_files)} recovery files → {recovery_manifest}",
          flush=True)

    # Estimate new positives from empirical rate
    prev_candidates = max(total_in_dedup - total_new, 1)
    est_pos_rate    = len(already_have) / prev_candidates
    est_pos_rate    = min(est_pos_rate, 0.10)   # cap at 10%
    est_new_pos     = int(total_new * est_pos_rate)

    print(f"\n{'='*65}")
    print(f"[prescan] RESULTS  ({dt/60:.1f} min,  {n_workers} workers)")
    print(f"{'='*65}")
    print(f"  .gz files scanned            : {len(gz_files):>12,}")
    print(f"  Records checked              : {total_checked:>12,}")
    print(f"  Pass structural filter       : {total_struct:>12,}  "
          f"({100*total_struct/max(total_checked,1):.1f}%)")
    print(f"  In dedup.tsv (real cands)    : {total_in_dedup:>12,}  "
          f"({100*total_in_dedup/max(total_struct,1):.1f}% of struct)")
    print(f"  Already in pass1_bert.jsonl  : {total_skip_have:>12,}")
    print(f"  NEW candidates for BERT      : {total_new:>12,}  ← tqdm total")
    print(f"  Files with missing candidates: {len(recovery_files):>12,}  "
          f"(out of {len(gz_files):,})")
    print(f"  Estimated new positives      : ~{est_new_pos:>10,}  "
          f"(empirical rate {est_pos_rate*100:.2f}%)")
    print(f"{'='*65}\n")

    return recovery_files, total_new


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: RECOVERY READER WORKER
# Identical to main pipeline reader_worker except:
#   - 5-gate logic (candidate_keys + already_have) instead of dedup writes
#   - No completed-files manifest (one-shot recovery operation)
#   - No dedup TSV writing (candidate_keys is read-only)
#   - Detailed per-gate skip counters for the final summary
# ─────────────────────────────────────────────────────────────────────────────

def recovery_reader_worker(
    worker_id,
    file_list,
    in_queue,
    cfg,
    chunk_size,
    year_start,
    year_end,
    candidate_keys,      # set[str] — read-only copy via spawn
    already_have,        # set[str] — read-only copy via spawn
    checked_val,         # Value(c_int64): total records scanned
    sent_val,            # Value(c_int64): records sent to BERT
    skipped_struct_val,  # Value(c_int64): rejected by structural filter
    skipped_cand_val,    # Value(c_int64): not in candidate_keys
    skipped_have_val,    # Value(c_int64): already in pass1_bert.jsonl
):
    try:
        # Per-worker dedup: files are partitioned non-overlapping across
        # workers so local_seen only catches rare within-worker duplicates
        # (same record appearing in multiple .gz files in one worker's list).
        local_seen = set()

        chunk_papers      = []
        chunk_texts       = []
        local_checked     = 0
        local_sent        = 0
        local_skip_struct = 0
        local_skip_cand   = 0
        local_skip_have   = 0

        def flush_counters():
            nonlocal local_checked, local_sent
            nonlocal local_skip_struct, local_skip_cand, local_skip_have
            with checked_val.get_lock():       checked_val.value       += local_checked
            with sent_val.get_lock():          sent_val.value          += local_sent
            with skipped_struct_val.get_lock():skipped_struct_val.value += local_skip_struct
            with skipped_cand_val.get_lock():  skipped_cand_val.value  += local_skip_cand
            with skipped_have_val.get_lock():  skipped_have_val.value  += local_skip_have
            local_checked     = 0
            local_sent        = 0
            local_skip_struct = 0
            local_skip_cand   = 0
            local_skip_have   = 0

        def flush_chunk():
            """Push chunk to GPU queue. Blocks until space — never drops records."""
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
                    print(f"[recover-reader-{worker_id}] Queue full, waiting ...",
                          flush=True)
                    time.sleep(5)

        for gz_path in file_list:
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

                        local_checked += 1

                        # Gate 1: structural filter — eliminates ~88% immediately
                        if not structural_filter(rec, year_start, year_end):
                            local_skip_struct += 1
                            if local_checked % COUNTER_FLUSH_INTERVAL == 0:
                                flush_counters()
                            continue

                        # Gate 2 + 3: key must exist AND be in dedup.tsv
                        key = make_key(rec)
                        if not key or key not in candidate_keys:
                            local_skip_cand += 1
                            if local_checked % COUNTER_FLUSH_INTERVAL == 0:
                                flush_counters()
                            continue

                        # Gate 4: must NOT already be in pass1_bert.jsonl
                        if key in already_have:
                            local_skip_have += 1
                            if local_checked % COUNTER_FLUSH_INTERVAL == 0:
                                flush_counters()
                            continue

                        # Gate 5: within-worker dedup
                        if key in local_seen:
                            if local_checked % COUNTER_FLUSH_INTERVAL == 0:
                                flush_counters()
                            continue
                        local_seen.add(key)

                        # All gates passed — build text and queue for GPU
                        text = (
                            build_text(rec, cfg)
                            or rec.get("display_name")
                            or rec.get("title")
                            or "unknown"
                        )
                        chunk_papers.append(slim(rec))
                        chunk_texts.append(text)
                        local_sent += 1

                        if local_checked % COUNTER_FLUSH_INTERVAL == 0:
                            flush_counters()
                        if len(chunk_papers) >= chunk_size:
                            flush_chunk()

            except Exception as e:
                print(f"[recover-reader-{worker_id}] ERROR on {gz_path}: {e}",
                      flush=True)
                traceback.print_exc()

        flush_chunk()
        flush_counters()

    except Exception as e:
        print(f"[recover-reader-{worker_id}] FATAL: {e}", flush=True)
        traceback.print_exc()
    finally:
        in_queue.put(None)  # sentinel — always sent so GPU worker can exit


# ─────────────────────────────────────────────────────────────────────────────
# GPU INFERENCE WORKER — identical to main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def gpu_worker(in_queue, out_queue, model_path, batch_size, device,
               max_len, num_readers, threshold):
    assert batch_size > 0,          "Batch size must be positive."
    assert 0.0 <= threshold <= 1.0, "Threshold must be 0.0–1.0."

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
                            "input_ids":
                                enc["input_ids"].to(device, non_blocking=True),
                            "attention_mask":
                                enc["attention_mask"].to(device, non_blocking=True),
                        }
                        if "token_type_ids" in enc:
                            inputs["token_type_ids"] = enc["token_type_ids"].to(
                                device, non_blocking=True)
                        logits = model(**inputs).logits.squeeze(-1)
                        probs_cpu[start:end] = (torch.sigmoid(logits)
                                                .float().cpu().numpy())
                        del inputs, logits, enc
                        if start % (batch_size * 4) == 0:
                            torch.cuda.empty_cache()

                    except torch.cuda.OutOfMemoryError:
                        print(f"[gpu:{device}] OOM at {start}-{end}, halving ...",
                              flush=True)
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
                            probs_cpu[ss:se] = (torch.sigmoid(logits)
                                                .float().cpu().numpy())
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

        out_queue.put(None)  # signal writer this GPU is done

    except Exception as e:
        print(f"\n[gpu:{device}] FATAL ERROR: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        out_queue.put(None)  # always send sentinel so writer can exit
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# WRITER WORKER
# Same as main pipeline except:
#   - WRITER_SYNC_INTERVAL = 500 (tighter crash safety)
#   - Tags each record with "recovery_pass": True
#   - Flushes pos_val after every batch to avoid undercounting on crash
# ─────────────────────────────────────────────────────────────────────────────

def writer_worker(out_queue, out_path, threshold, pos_val, num_gpu_workers):
    try:
        unflushed_pos = 0
        done_count    = 0

        with open(out_path, "a", encoding="utf-8", buffering=8 * 1024 * 1024) as f:
            while done_count < num_gpu_workers:
                try:
                    item = out_queue.get(timeout=WRITER_QUEUE_TIMEOUT_S)
                except Exception:
                    # DEBUG: no GPU output for WRITER_QUEUE_TIMEOUT_S seconds
                    with pos_val.get_lock():
                        current_pos = pos_val.value
                    print(
                        f"[writer] {WRITER_QUEUE_TIMEOUT_S}s timeout "
                        f"({done_count}/{num_gpu_workers} sentinels received). "
                        f"Current positives so far = {current_pos}. "
                        f"No batches from GPUs in the last {WRITER_QUEUE_TIMEOUT_S}s; "
                        f"GPUs may be slow or all scores < threshold. "
                        f"Continuing to wait...",
                        flush=True,
                    )
                    # IMPORTANT: do NOT break here; just keep waiting
                    continue

                if item is None:
                    done_count += 1
                    continue

                batch_pos = 0
                for paper, prob in item:
                    batch_pos     += 1
                    unflushed_pos += 1
                    out = dict(paper)
                    out.update({
                        "bert_score":     prob,
                        "bert_label":     1,
                        "bert_threshold": threshold,
                        "recovery_pass":  True,
                    })
                    f.write(orjson.dumps(out).decode("utf-8") + "\n")

                # Flush pos_val after every batch — no undercounting on crash
                if batch_pos > 0:
                    with pos_val.get_lock():
                        pos_val.value += batch_pos

                if unflushed_pos >= WRITER_SYNC_INTERVAL:
                    f.flush()
                    os.fsync(f.fileno())
                    unflushed_pos = 0

            # Final flush on any exit path
            f.flush()
            os.fsync(f.fileno())

    except Exception as e:
        print(f"\n[writer] FATAL ERROR: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 LAUNCHER
# Shared by both prescan path and --skip_prescan path.
# All parameters passed explicitly — no dynamic attributes.
# ─────────────────────────────────────────────────────────────────────────────

def run_phase2(
    files_to_scan:        list,
    total_new_candidates: int,    # exact count from prescan, or -1 if unknown
    candidate_keys:       set,
    already_have:         set,
    gpu_configs:          List[GpuConfig],
    model_path:           str,    # passed explicitly — no dynamic attributes
    max_len:              int,    # passed explicitly — no dynamic attributes
    cfg:                  dict,
    out_jsonl:            str,
    threshold:            float,
    year_start:           int,
    year_end:             int,
    seed:                 int,
    all_gz_count:         int,    # total .gz files (for display only)
):
    total_readers = sum(g.num_readers for g in gpu_configs)

    # Backup pass1_bert.jsonl before any writes
    if os.path.exists(out_jsonl):
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = f"{out_jsonl}.recovery_backup.{ts}"
        shutil.copy2(out_jsonl, bak)
        size_mb = os.path.getsize(bak) / 1e6
        print(f"\n[recovery] Backed up pass1_bert.jsonl "
              f"→ {os.path.basename(bak)}  ({size_mb:.1f} MB)")

    # Partition recovery files across GPU reader workers
    partitions = partition_files_gpu(files_to_scan, gpu_configs, seed=seed)

    # Shared counters
    checked_val        = Value(ctypes.c_int64, 0)
    sent_val           = Value(ctypes.c_int64, 0)
    pos_val            = Value(ctypes.c_int64, 0)
    skipped_struct_val = Value(ctypes.c_int64, 0)
    skipped_cand_val   = Value(ctypes.c_int64, 0)
    skipped_have_val   = Value(ctypes.c_int64, 0)

    # Build queues, GPU workers, reader workers
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
            args=(
                iq, out_queue,
                model_path,        # passed as plain argument
                gcfg.batch_size,
                dev,
                max_len,           # passed as plain argument
                gcfg.num_readers,
                threshold,
            ),
            daemon=True,
        )
        gp.start()
        gpu_procs.append(gp)
        print(f"[main] GPU worker on {dev} "
              f"(batch={gcfg.batch_size}, readers={gcfg.num_readers})", flush=True)

        for ri in range(gcfg.num_readers):
            wid       = global_reader_id
            file_list = partitions.get((gi, ri), [])
            p = Process(
                target=recovery_reader_worker,
                args=(
                    wid, file_list, iq, cfg, gcfg.chunk_size,
                    year_start, year_end,
                    candidate_keys, already_have,
                    checked_val, sent_val,
                    skipped_struct_val, skipped_cand_val, skipped_have_val,
                ),
                daemon=True,
            )
            p.start()
            readers.append(p)
            print(f"[main] recover-reader-{wid} → {dev}  "
                  f"({len(file_list)} files)", flush=True)
            global_reader_id += 1

    # Free parent-process copies — children have their own via spawn
    del candidate_keys, already_have

    writer_proc = Process(
        target=writer_worker,
        args=(out_queue, out_jsonl, threshold, pos_val, len(gpu_configs)),
        daemon=True,
    )
    writer_proc.start()

    # Signal handlers
    def kill_all():
        try:
            for iq in in_queues:
                while not iq.empty(): iq.get_nowait()
            while not out_queue.empty(): out_queue.get_nowait()
        except Exception:
            pass
        for iq in in_queues:
            iq.cancel_join_thread()
        out_queue.cancel_join_thread()
        for p in readers + gpu_procs + [writer_proc]:
            if p.is_alive(): p.terminate()
        time.sleep(1)
        for p in readers + gpu_procs + [writer_proc]:
            if p.is_alive(): p.kill()

    def _shutdown(sig, frame):
        print(f"\n[signal {sig}] Shutting down cleanly ...", flush=True)
        kill_all()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    # Startup summary
    # When total_new_candidates >= 0 (prescan ran): tqdm has total → real ETA
    # When total_new_candidates == -1 (--skip_prescan): no total → running count
    tqdm_total = total_new_candidates if total_new_candidates >= 0 else None

    print(f"\n[{datetime.now():%H:%M:%S}] Recovery pipeline started")
    print(f"  Files to re-scan           : {len(files_to_scan):>12,}  "
          f"(of {all_gz_count:,} total .gz files)")
    if tqdm_total is not None:
        print(f"  BERT candidates (prescan)  : {tqdm_total:>12,}  ← tqdm total / ETA")
    else:
        print(f"  BERT candidates            :      unknown  "
              f"(--skip_prescan: bar shows running sent count)")
    print(f"  Total reader workers       : {total_readers}")
    print(f"  GPU workers                : {len(gpu_configs)}")
    print(f"  BERT threshold             : {threshold}")
    print(f"  Writer sync every          : {WRITER_SYNC_INTERVAL} positives")
    for gi, gc in enumerate(gpu_configs):
        dev = gc.device if torch.cuda.is_available() else "cpu"
        print(f"    GPU {gi}: {dev}  batch={gc.batch_size}  chunk={gc.chunk_size}  "
              f"queue={gc.queue_max}  readers={gc.num_readers}")
    print()

    # Monitoring loop
    t0, last_sent = time.time(), 0
    pbar = tqdm(
        total=tqdm_total,
        desc="Recovering",
        unit="cand",
        dynamic_ncols=True,
        mininterval=2.0,
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            if tqdm_total is not None else
            "{desc}: {n_fmt} sent [{elapsed}, {rate_fmt}] {postfix}"
        ),
    )

    while True:
        time.sleep(5)

        checked     = checked_val.value
        sent        = sent_val.value
        pos         = pos_val.value
        skip_struct = skipped_struct_val.value
        skip_cand   = skipped_cand_val.value
        skip_have   = skipped_have_val.value
        elapsed     = max(time.time() - t0, 1)

        delta = sent - last_sent
        if delta > 0:
            pbar.update(delta)
        last_sent = sent

        pbar.set_postfix({
            "scanned":     f"{checked/1e6:.2f}M",
            "skip_struct": f"{skip_struct/1e6:.2f}M",
            "skip_cand":   f"{skip_cand/1e6:.2f}M",
            "skip_have":   f"{skip_have/1e3:.1f}k",
            "sent_BERT":   f"{sent/1e3:.1f}k",
            "NEW+":        f"{pos:,}",
            "rec/s":       f"{checked/elapsed:,.0f}",
            "iq":          "+".join(str(iq.qsize()) for iq in in_queues),
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
            print("\n[FATAL] Writer crashed. Shutting down.", flush=True)
            kill_all()
            sys.exit(1)

    pbar.close()
    for p in readers:   p.join(timeout=60)
    for p in gpu_procs: p.join(timeout=60)
    writer_proc.join(timeout=60)

    # Final summary
    dt      = time.time() - t0
    checked = checked_val.value
    sent    = sent_val.value
    pos     = pos_val.value
    s_struct= skipped_struct_val.value
    s_cand  = skipped_cand_val.value
    s_have  = skipped_have_val.value

    print("=" * 65)
    print(f"[{datetime.now():%H:%M:%S}] RECOVERY COMPLETE")
    print(f"  Files re-scanned           : {len(files_to_scan):>12,}")
    print(f"  Records scanned            : {checked:>12,}")
    print(f"  |-  skip structural        : {s_struct:>12,}  "
          f"({100*s_struct/max(checked,1):.1f}%)")
    print(f"  |-  skip not in dedup.tsv  : {s_cand:>12,}")
    print(f"  |-  skip already written   : {s_have:>12,}")
    print(f"  +-  sent to BERT           : {sent:>12,}")
    print(f"  NEW positives recovered    : {pos:>12,}  (threshold={threshold})")
    print(f"  Time                       : {dt/60:>11.1f} min")
    print(f"  Throughput                 : {checked/max(dt,1):>10,.0f} rec/s")
    print(f"  Output appended to         : {out_jsonl}")
    print("=" * 65)
    print()
    print("  Recovered papers tagged  \"recovery_pass\": true  in JSONL.")
    print("  Recommended: final dedup on pass1_bert.jsonl by DOI/OAID")
    print("  before downstream use.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--snapshot_dir",    required=True,
                    help="Directory containing OpenAlex .gz files.")
    ap.add_argument("--output_dir",      required=True,
                    help="Same output_dir as main pipeline. "
                         "Must contain dedup.tsv.")
    ap.add_argument("--model",           default=None,
                    help="BERT model path. Not needed for --prescan-only.")
    ap.add_argument("--config",          default="bert_classifier_config.yaml")
    ap.add_argument("--threshold",       type=float, default=None)
    ap.add_argument("--year_start",      type=int,   default=2000)
    ap.add_argument("--year_end",        type=int,   default=2026)
    ap.add_argument("--pattern",         default="**/*.gz")
    ap.add_argument("--seed",            type=int,   default=42,
                    help="RNG seed for file partitioning.")
    ap.add_argument("--prescan_workers", type=int,   default=8,
                    help="Parallel CPU workers for prescan. Default 8.")
    ap.add_argument(
        "--gpu_configs", nargs="+",
        default=["0:batch=1024,chunk=16384,queue=4,readers=2"],
        metavar="GPU_SPEC",
        help='Per-GPU config. Example: "0:batch=1024,chunk=16384,queue=4,readers=2"',
    )
    ap.add_argument("--prescan-only", action="store_true",
                    help="Run prescan only, write recovery_files.txt, then exit.")
    ap.add_argument("--skip_prescan", action="store_true",
                    help=(
                        "Skip Phase 1. Go straight to Phase 2 on ALL files. "
                        "Use when missing candidates are spread across the full "
                        "dataset (dedup sync happened across many files at crash "
                        "time). Progress bar shows running sent count with no ETA."
                    ))
    args = ap.parse_args()

    # ── Mutual exclusion ──────────────────────────────────────────────────────
    if args.prescan_only and args.skip_prescan:
        ap.error("--prescan-only and --skip_prescan are mutually exclusive.")

    # ── Load YAML config ──────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    threshold = (args.threshold if args.threshold is not None
                 else float(cfg["inference"]["threshold"]))
    max_len   = int(cfg["model"]["max_len"])

    out_jsonl = os.path.join(args.output_dir, "pass1_bert.jsonl")
    dedup_tsv = os.path.join(args.output_dir, "dedup.tsv")

    # ── Find .gz files ────────────────────────────────────────────────────────
    gz_files = sorted(
        glob(os.path.join(args.snapshot_dir, args.pattern), recursive=True),
        key=os.path.getsize, reverse=True,
    )
    gz_files = [f for f in gz_files if f.endswith(".gz")]
    if not gz_files:
        raise FileNotFoundError(f"No .gz files found under {args.snapshot_dir}.")

    # ── Phase 0: Load state ───────────────────────────────────────────────────
    candidate_keys = load_candidate_keys(dedup_tsv)
    already_have   = load_already_have(out_jsonl)

    print(f"\n[recovery] State loaded:")
    print(f"  candidate_keys (dedup.tsv)  : {len(candidate_keys):>12,}")
    print(f"  already_have (pass1_bert)   : {len(already_have):>12,}")
    print(f"  .gz files to scan           : {len(gz_files):>12,}")
    print(f"  BERT threshold              : {threshold}")
    print(f"  Year range                  : {args.year_start}–{args.year_end}")
    if args.skip_prescan:
        print(f"  Prescan                     : SKIPPED (--skip_prescan)")
    else:
        print(f"  Prescan workers             : {args.prescan_workers}")
    print()

    # ── Prescan-only mode ─────────────────────────────────────────────────────
    if args.prescan_only:
        prescan_parallel(
            gz_files       = gz_files,
            candidate_keys = candidate_keys,
            already_have   = already_have,
            year_start     = args.year_start,
            year_end       = args.year_end,
            n_workers      = args.prescan_workers,
            seed           = args.seed,
            output_dir     = args.output_dir,
        )
        print("[prescan-only] Done. recovery_files.txt written. No GPU launched.")
        return

    # ── Validate model ────────────────────────────────────────────────────────
    if not args.model:
        ap.error("--model is required unless --prescan-only is set.")

    # ── Parse and validate GPU configs ────────────────────────────────────────
    gpu_configs: List[GpuConfig] = [parse_gpu_config(s) for s in args.gpu_configs]

    if torch.cuda.is_available():
        n_avail = torch.cuda.device_count()
        print(f"[gpu] {n_avail} CUDA device(s) visible:")
        for gc in gpu_configs:
            if gc.device_id >= n_avail:
                raise ValueError(
                    f"Requested cuda:{gc.device_id} but only {n_avail} visible. "
                    f"Check CUDA_VISIBLE_DEVICES."
                )
            free_mem, total_mem = torch.cuda.mem_get_info(gc.device_id)
            print(f"  {gc}  |  free={free_mem/1e9:.1f}/{total_mem/1e9:.1f} GB")
    else:
        print("WARNING: No CUDA GPUs found — running on CPU (very slow).")
        gpu_configs = [GpuConfig(device_id=-1, batch_size=64, chunk_size=512,
                                 queue_max=2, num_readers=1)]

    # ── Determine which files to scan and candidate count ─────────────────────
    if args.skip_prescan:
        files_to_scan        = gz_files
        total_new_candidates = -1   # unknown — tqdm has no total
        print(f"[recovery] --skip_prescan: scanning all {len(gz_files):,} files.")
        print(f"[recovery] Progress bar shows running sent count (no ETA).\n")
    else:
        recovery_files, total_new_candidates = prescan_parallel(
            gz_files       = gz_files,
            candidate_keys = candidate_keys,
            already_have   = already_have,
            year_start     = args.year_start,
            year_end       = args.year_end,
            n_workers      = args.prescan_workers,
            seed           = args.seed,
            output_dir     = args.output_dir,
        )
        if total_new_candidates == 0:
            print("[recovery] Prescan found 0 new candidates. Nothing to recover.")
            return
        files_to_scan = recovery_files

    # ── Phase 2: Recovery inference ───────────────────────────────────────────
    run_phase2(
        files_to_scan        = files_to_scan,
        total_new_candidates = total_new_candidates,
        candidate_keys       = candidate_keys,
        already_have         = already_have,
        gpu_configs          = gpu_configs,
        model_path           = args.model,    # plain argument — no dynamic attrs
        max_len              = max_len,        # plain argument — no dynamic attrs
        cfg                  = cfg,
        out_jsonl            = out_jsonl,
        threshold            = threshold,
        year_start           = args.year_start,
        year_end             = args.year_end,
        seed                 = args.seed,
        all_gz_count         = len(gz_files),
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()


# ─────────────────────────────────────────────────────────────────────────────
# NOTE ON RAM: BLOOM FILTER FOR candidate_keys
# ─────────────────────────────────────────────────────────────────────────────
# 100M keys as Python set ≈ 5–8 GB RAM.
# If OOM, replace load_candidate_keys() with:
#
#   pip install pybloom-live
#   from pybloom_live import ScalableBloomFilter
#   bf = ScalableBloomFilter(mode=ScalableBloomFilter.LARGE_SET_GROWTH)
#   for key in tsv_keys: bf.add(key)
#
# ~300 MB at 0.1% false positive rate. False positives only send ~0.1% of
# non-candidates to BERT — harmless, BERT rejects them anyway.
