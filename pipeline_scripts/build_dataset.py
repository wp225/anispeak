#!/usr/bin/env python3
"""
build_bert_dataset.py  v6
=========================
Config-driven. Reads bert_classifier_config.yaml.

Changes from v5:
  - Positives:  score >= min_score (7) only — clean high-confidence labels
  - Negatives:  [D] score <= hard_neg_max_score (3) from final_relevant  (easy, clean)
                [A] Qwen-rejected (is_relevant=False, passed Mistral)     (hard negatives)
                [B] Mistral high-conf rejected (conf>=0.85)               (bulk fill to 1:1)
                [C] Blacklisted heuristic                                 (last resort)
  - score=5 papers COMPLETELY EXCLUDED — Qwen was uncertain, label noise both ways
  - 1:1 pos:neg ratio (neg_ratio: 1.0 in config)
  - Dedup by DOI + normalized title (shared set across ALL sources)
  - tqdm on every loop + file scan
  - ThreadPoolExecutor for build_text (CPU-bound, stateless, safe to thread)
  - Dedup stays in main thread (sets are NOT thread-safe)
  - load_jsonl pre-counts lines for accurate tqdm total
  - split_and_save writes 3 splits in parallel threads
  - Stats file includes per-source breakdown + dedup counts + score distribution
  - tqdm.write() used for all logging (thread-safe, no bar corruption)

Usage:
  python build_bert_dataset.py --config bert_classifier_config.yaml
"""

import argparse
import json
import os
import random
import re
import sqlite3
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    print("pip install tqdm"); raise

try:
    import yaml
except ImportError:
    print("pip install pyyaml"); raise


# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg: str):
    import datetime
    tqdm.write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")


# ── Deduplication helpers ─────────────────────────────────────────────────────

def normalize_doi(doi: str) -> str:
    if not doi: return ""
    d = doi.lower().strip()
    for prefix in ("https://doi.org/", "http://doi.org/",
                   "https://dx.doi.org/", "http://dx.doi.org/"):
        d = d.replace(prefix, "")
    return d.strip()


def normalize_title(title: str) -> str:
    if not title: return ""
    t = title.lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def get_dedup_keys(paper: dict) -> list:
    """
    Returns dedup keys for this paper.
    A paper is a duplicate if ANY of its keys are already seen.
    doi:...   — most reliable; same DOI = definitively same paper
    title:... — fallback; catches same paper re-fetched from multiple batches
    """
    keys = []
    doi = normalize_doi(paper.get("doi") or "")
    if doi:
        keys.append(f"doi:{doi}")
    title = normalize_title(paper.get("title") or paper.get("display_name") or "")
    if title and len(title) > 10:   # skip very short titles to avoid false positives
        keys.append(f"title:{title}")
    return keys


# ── OpenAlex helpers ──────────────────────────────────────────────────────────

def get_oaid(paper: dict):
    m = re.search(r"W\d+", str(paper.get("id") or paper.get("oaid") or ""))
    return m.group() if m else None


def reconstruct_abstract(inv) -> str:
    if not inv or not isinstance(inv, dict): return ""
    try:
        pairs = [(pos, w) for w, positions in inv.items() for pos in positions]
        pairs.sort()
        return " ".join(w for _, w in pairs)
    except Exception:
        return ""


# ── build_text — SINGLE SOURCE OF TRUTH ──────────────────────────────────────
# Imported by train_bert_classifier.py.
# Any change here automatically propagates to dataset building AND inference.

def _safe_str(x):
    """Robust string extractor for noisy OpenAlex fields."""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("display_name", "name", "label", "title"):
            v = x.get(k)
            if isinstance(v, str):
                return v
    # Lists / tuples → join any strings
    if isinstance(x, (list, tuple)):
        parts = [p for p in x if isinstance(p, str)]
        if parts:
            return "; ".join(parts)
    return ""


def build_text(paper: dict, cfg: dict) -> str:
    f = cfg.get("input_fields", {})

    parts = []

    # TITLE
    if f.get("use_title", True):
        t = (paper.get("title") or paper.get("display_name") or "").strip()
        if t:
            parts.append(f"[TITLE] {t}")

    # VENUE (unchanged)
    if f.get("use_venue", True):
        v = (paper.get("venue") or "").strip()
        if v:
            parts.append(f"[VENUE] {v}")

    # PRIMARY TOPIC FIELD — robust handling
    if f.get("use_primary_topic_field", True):
        pt = paper.get("primary_topic")
        fn = ""
        if isinstance(pt, dict):
            # pt may look like:
            # {"field": "Bioacoustics", ...}
            # or {"field": {"display_name": "Bioacoustics", ...}, ...}
            raw = pt.get("field") or pt.get("subfield") or ""
            fn  = _safe_str(raw).strip()
        elif isinstance(pt, str):
            fn = pt.strip()
        if fn:
            parts.append(f"[FIELD] {fn}")

    # KEYWORDS (unchanged but robust to dict/list)
    if f.get("use_keywords", True):
        ks = []
        for k in (paper.get("keywords") or [])[: f.get("max_keywords", 8)]:
            if isinstance(k, dict):
                val = _safe_str(k)
            else:
                val = str(k)
            val = val.strip()
            if val:
                ks.append(val)
        if ks:
            parts.append(f"[KEYWORDS] {'; '.join(ks)}")

    # ABSTRACT (unchanged)
    if f.get("use_abstract", True):
        ab = (paper.get("abstract") or "").strip()
        if not ab:
            ab = reconstruct_abstract(paper.get("abstract_inverted_index"))
        if ab:
            parts.append(f"[ABSTRACT] {ab}")

    return " ".join(parts)

# ── Parallel text building ────────────────────────────────────────────────────

def _build_one(args):
    """
    Stateless worker — safe for ThreadPoolExecutor.
    Returns example dict with '_paper' key (for dedup in main thread).
    Returns None if text is too short (filtered out).
    """
    paper, label, source, score, cfg, min_text_len = args
    text = build_text(paper, cfg)
    if not text or len(text) < min_text_len:
        return None
    return {
        "id":              get_oaid(paper) or paper.get("doi") or "",
        "text":            text,
        "title":           (paper.get("title") or paper.get("display_name") or "").strip(),
        "label":           label,
        "source":          source,
        "relevance_score": score,
        "_paper":          paper,   # stripped before saving — used only for dedup
    }


def parallel_build_examples(
    papers_with_meta: list,
    cfg: dict,
    num_workers: int,
    desc: str = "Building",
) -> list:
    """
    Build examples in parallel using ThreadPoolExecutor.

    Why ThreadPoolExecutor (NOT ProcessPoolExecutor):
      - build_text() is pure Python string ops — no I/O, no GPU
      - ProcessPoolExecutor would pickle every paper dict (expensive for 20k+ papers)
      - Threads share memory directly — no serialization overhead
      - ~3-5x speedup vs sequential for large batches

    CRITICAL: Dedup (seen_keys/seen_oids) MUST happen AFTER in the main thread.
    Python sets are NOT thread-safe.
    """
    min_len   = cfg["dataset"].get("min_text_len", 30)
    args_list = [(p, l, s, sc, cfg, min_len) for p, l, s, sc in papers_with_meta]
    results   = [None] * len(args_list)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_build_one, a): i for i, a in enumerate(args_list)}
        pbar = tqdm(as_completed(futures), total=len(futures), desc=desc,
                    unit="papers", dynamic_ncols=True)
        for future in pbar:
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = None

    return [r for r in results if r is not None]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path: str, desc: str = None) -> list:
    """Load JSONL with tqdm. Pre-counts lines for accurate progress total."""
    records = []
    if not os.path.exists(path):
        log(f"  WARNING: file not found, skipping: {path}")
        return records
    with open(path, "rb") as f:
        total = sum(1 for _ in f)
    label = desc or os.path.basename(path)
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, total=total, desc=f"  Loading {label}",
                         unit="lines", dynamic_ncols=True):
            line = line.strip()
            if not line: continue
            try: records.append(json.loads(line))
            except Exception: continue
    log(f"  Loaded {len(records):,} records from {os.path.basename(path)}")
    return records


def load_stage_db(db_path: str):
    """
    Load Stage-1 (Mistral) and Stage-2 (Qwen) results from SQLite.

    Actual table/column names from classify_papers.py ClassifyDB._create_tables():
      processed_s1:  oa_id, relevant, confidence, reason, processed_at
      processed_s2:  oa_id, is_relevant, relevance_score, paper_type, summary, processed_at
    """
    if not os.path.exists(db_path):
        log(f"  WARNING: classify.sqlite not found: {db_path}")
        return {}, {}

    log(f"  Reading SQLite: {os.path.basename(db_path)}")
    conn = sqlite3.connect(db_path)

    # Stage 2 — Qwen results
    s2 = {
        r[0]: {"is_relevant": bool(r[1]), "relevance_score": int(r[2] or 0)}
        for r in conn.execute(
            "SELECT oa_id, is_relevant, relevance_score FROM processed_s2"
        ).fetchall()
    }

    # Stage 1 — Mistral results
    s1 = {
        r[0]: {"relevant": bool(r[1]), "confidence": float(r[2] or 0.0)}
        for r in conn.execute(
            "SELECT oa_id, relevant, confidence FROM processed_s1"
        ).fetchall()
    }

    conn.close()
    log(f"  DB loaded — Stage-1: {len(s1):,}  Stage-2: {len(s2):,}")
    return s1, s2


def split_and_save(examples: list, cfg: dict) -> dict:
    """Shuffle, split train/val/test, write in parallel threads."""
    dataset_dir = cfg["paths"]["dataset_dir"]
    rng = random.Random(cfg["dataset"]["seed"])
    rng.shuffle(examples)
    n       = len(examples)
    n_test  = int(n * cfg["dataset"]["test_ratio"])
    n_val   = int(n * cfg["dataset"]["val_ratio"])
    n_train = n - n_test - n_val
    splits  = {
        "train": examples[:n_train],
        "val":   examples[n_train: n_train + n_val],
        "test":  examples[n_train + n_val:],
    }
    os.makedirs(dataset_dir, exist_ok=True)

    def _write(name_records):
        name, records = name_records
        path = os.path.join(dataset_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for rec in tqdm(records, desc=f"  Writing {name}", unit="rec",
                            dynamic_ncols=True, leave=False):
                # Strip _paper key — it was for dedup only, not needed in training data
                out = {k: v for k, v in rec.items() if k != "_paper"}
                f.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
        pos = sum(1 for r in records if r["label"] == 1)
        log(f"  {name:5s}: {len(records):6,}  pos={pos:,}  neg={len(records)-pos:,}  -> {path}")

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(_write, splits.items()))
    return splits


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="bert_classifier_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pipeline_dir       = cfg["paths"]["pipeline_dir"]
    fetch_jsonl        = cfg["paths"]["fetch_jsonl"]
    db_path            = os.path.join(pipeline_dir, "classify.sqlite")
    dcfg               = cfg["dataset"]
    seed               = dcfg["seed"]
    num_workers        = int(dcfg.get("num_workers", os.cpu_count() or 4))
    min_score          = int(dcfg["min_score"])
    hard_neg_max_score = int(dcfg.get("hard_neg_max_score", 3))
    neg_ratio          = float(dcfg.get("neg_ratio", 1.0))
    max_neg            = int(dcfg.get("max_neg", 100_000))

    # Sanity check: excluded gap must exist between neg and pos thresholds
    if hard_neg_max_score >= min_score:
        log(f"ERROR: hard_neg_max_score ({hard_neg_max_score}) must be < "
            f"min_score ({min_score}). Score=5 must fall in the excluded gap. Aborting.")
        return

    log("=" * 65)
    log("BERT Dataset Builder v6  (clean labels — no ambiguous score=5)")
    log(f"Config      : {args.config}")
    log(f"Model       : {cfg['model']['name']}")
    log(f"num_workers : {num_workers}")
    log(f"Positives   : score >= {min_score}")
    log(f"Neg-D       : score <= {hard_neg_max_score} from final_relevant")
    log(f"EXCLUDED    : score {hard_neg_max_score+1}–{min_score-1} (Qwen uncertain — skipped)")
    log(f"Neg-B fill  : Mistral high-conf rejected (conf>=0.85)")
    log(f"Ratio       : 1:{neg_ratio:.1f}")
    fields_on = [k.replace("use_", "") for k, v in cfg["input_fields"].items()
                 if k.startswith("use_") and v]
    log(f"Fields      : {', '.join(fields_on)}")
    log("=" * 65)

    s1, s2 = load_stage_db(db_path)

    # ── Global dedup — MAIN THREAD ONLY ──────────────────────────────────────
    seen_keys: set  = set()
    seen_oids: set  = set()
    dedup_stats     = defaultdict(int)

    def is_duplicate(paper: dict, source: str) -> bool:
        oid  = get_oaid(paper)
        if oid and oid in seen_oids:
            dedup_stats[f"{source}:oaid_dup"] += 1; return True
        keys = get_dedup_keys(paper)
        if any(k in seen_keys for k in keys):
            dedup_stats[f"{source}:doi_title_dup"] += 1; return True
        if oid: seen_oids.add(oid)
        for k in keys: seen_keys.add(k)
        return False

    def get_score(paper: dict, oid) -> int:
        if oid and oid in s2: return s2[oid]["relevance_score"]
        return int(paper.get("relevance_score") or paper.get("relevancescore") or 0)

    # ── Load final_relevant.jsonl ONCE (used for positives + neg-D) ───────────
    log("Loading final_relevant.jsonl ...")
    final_rel = load_jsonl(
        os.path.join(pipeline_dir, "final_relevant.jsonl"), "final_relevant"
    )
    if not final_rel:
        log("ERROR: final_relevant.jsonl is empty or missing. Aborting."); return

    # Print score distribution before building
    score_dist = Counter(get_score(p, get_oaid(p)) for p in final_rel)
    log(f"Score distribution ({len(final_rel):,} raw papers):")
    for s in sorted(score_dist):
        tag = "→ POSITIVE" if s >= min_score else (
              "→ EXCLUDED" if s > hard_neg_max_score else "→ NEG-D")
        log(f"    score={s}: {score_dist[s]:>7,}  {tag}")

    # ── POSITIVES ─────────────────────────────────────────────────────────────
    log(f"\nBuilding POSITIVES (score >= {min_score}) ...")
    pos_candidates = []
    for paper in tqdm(final_rel, desc="  Score-filtering positives",
                      unit="papers", dynamic_ncols=True):
        if not paper.get("title"): continue
        oid = get_oaid(paper); sc = get_score(paper, oid)
        if sc >= min_score:
            pos_candidates.append((paper, 1, "stage2_verified", sc))

    log(f"  Candidates: {len(pos_candidates):,}")
    pos_built = parallel_build_examples(pos_candidates, cfg, num_workers, "  Building positives")

    positives: list = []
    for ex in tqdm(pos_built, desc="  Deduping positives", unit="papers", dynamic_ncols=True):
        if not is_duplicate(ex["_paper"], "pos"):
            positives.append(ex)

    log(f"  ✓ Positives after dedup: {len(positives):,}  "
        f"(deduped {len(pos_built) - len(positives):,})")
    if not positives:
        log("ERROR: No positives found. Check pipeline_dir and min_score."); return

    # ── NEGATIVES ─────────────────────────────────────────────────────────────
    target_neg = min(int(len(positives) * neg_ratio), max_neg)
    log(f"\nBuilding NEGATIVES (target={target_neg:,}, ratio={neg_ratio:.1f}x) ...")
    negatives: list = []

    # [D] score<=3 from final_relevant (already in memory — no extra file read)
    log(f"  [D] Qwen score<={hard_neg_max_score} from final_relevant (clean negatives) ...")
    neg_D_cands = []
    for paper in tqdm(final_rel, desc="  Gathering neg-D", unit="papers", dynamic_ncols=True):
        if not paper.get("title"):
            dedup_stats["neg_D:no_title"] += 1; continue
        oid = get_oaid(paper); sc = get_score(paper, oid)
        if sc > hard_neg_max_score: continue   # skip score=5 (excluded) and score>=7 (positives)
        neg_D_cands.append((paper, 0, f"qwen_score{sc}_neg", sc))

    log(f"    Candidates: {len(neg_D_cands):,}")
    neg_D_built = parallel_build_examples(neg_D_cands, cfg, num_workers, "  Building neg-D")
    for ex in tqdm(neg_D_built, desc="  Deduping neg-D", unit="papers", dynamic_ncols=True):
        if len(negatives) >= target_neg: break
        if not is_duplicate(ex["_paper"], "neg_D"): negatives.append(ex)
    log(f"    After [D]: {len(negatives):,}  (gap={max(0, target_neg-len(negatives)):,})")

    # [A] Qwen-rejected (is_relevant=False)
    if len(negatives) < target_neg:
        log(f"  [A] Qwen-rejected (passed Mistral, failed Stage 2) ...")
        pass1 = load_jsonl(os.path.join(pipeline_dir, "pass1_relevant.jsonl"), "pass1_relevant")
        pass1_by_oid = {get_oaid(p): p for p in pass1 if get_oaid(p)}

        neg_A_cands = []
        for oid, info in tqdm(s2.items(), desc="  Gathering neg-A", dynamic_ncols=True):
            if info["is_relevant"]: continue
            paper = pass1_by_oid.get(oid)
            if not paper or not paper.get("title"):
                dedup_stats["neg_A:no_title"] += 1; continue
            neg_A_cands.append((paper, 0, "stage2_qwen_rejected", 0))

        log(f"    Candidates: {len(neg_A_cands):,}")
        neg_A_built = parallel_build_examples(neg_A_cands, cfg, num_workers, "  Building neg-A")
        for ex in tqdm(neg_A_built, desc="  Deduping neg-A", unit="papers", dynamic_ncols=True):
            if len(negatives) >= target_neg: break
            if not is_duplicate(ex["_paper"], "neg_A"): negatives.append(ex)
        log(f"    After [A]: {len(negatives):,}  (gap={max(0, target_neg-len(negatives)):,})")

    # [B] Mistral high-confidence rejected (bulk filler)
    if len(negatives) < target_neg:
        still_need = target_neg - len(negatives)
        log(f"  [B] Mistral high-conf rejected (conf>=0.85) — need {still_need:,} ...")
        need_oids = set(list({
            oid for oid, info in s1.items()
            if not info["relevant"] and info.get("confidence", 0.0) >= 0.85
        })[: still_need + 20_000])
        log(f"    Available OAIDs: {len(need_oids):,}")

        if need_oids and os.path.exists(fetch_jsonl):
            with open(fetch_jsonl, "rb") as f:
                total_lines = sum(1 for _ in f)
            neg_B_cands = []
            with open(fetch_jsonl, encoding="utf-8") as f:
                pbar = tqdm(f, total=total_lines, desc="  Scanning fetch_jsonl [B]",
                            unit="lines", dynamic_ncols=True)
                for line in pbar:
                    line = line.strip()
                    if not line: continue
                    try: paper = json.loads(line)
                    except Exception: continue
                    oid = get_oaid(paper)
                    if oid not in need_oids: continue
                    need_oids.discard(oid)
                    if not paper.get("title"):
                        dedup_stats["neg_B:no_title"] += 1; continue
                    neg_B_cands.append((paper, 0, "stage1_mistral_rejected", 0))
                    pbar.set_postfix({"found": len(neg_B_cands), "remaining": len(need_oids)})
                    if not need_oids: break

            log(f"    Candidates: {len(neg_B_cands):,}")
            neg_B_built = parallel_build_examples(neg_B_cands, cfg, num_workers, "  Building neg-B")
            for ex in tqdm(neg_B_built, desc="  Deduping neg-B", unit="papers", dynamic_ncols=True):
                if len(negatives) >= target_neg: break
                if not is_duplicate(ex["_paper"], "neg_B"): negatives.append(ex)
        log(f"    After [B]: {len(negatives):,}  (gap={max(0, target_neg-len(negatives)):,})")

    # [C] Blacklisted heuristic — last resort
    if len(negatives) < target_neg and os.path.exists(fetch_jsonl):
        still_need = target_neg - len(negatives)
        log(f"  [C] Blacklisted heuristic — last resort (need {still_need:,}) ...")
        with open(fetch_jsonl, "rb") as f:
            total_lines = sum(1 for _ in f)
        neg_C_cands = []
        with open(fetch_jsonl, encoding="utf-8") as f:
            for line in tqdm(f, total=total_lines, desc="  Scanning fetch_jsonl [C]",
                             unit="lines", dynamic_ncols=True):
                line = line.strip()
                if not line: continue
                try: paper = json.loads(line)
                except Exception: continue
                if not paper.get("blacklisted", False): continue
                if not paper.get("title"):
                    dedup_stats["neg_C:no_title"] += 1; continue
                neg_C_cands.append((paper, 0, "heuristic_blacklisted", 0))

        log(f"    Candidates: {len(neg_C_cands):,}")
        neg_C_built = parallel_build_examples(neg_C_cands, cfg, num_workers, "  Building neg-C")
        for ex in tqdm(neg_C_built, desc="  Deduping neg-C", unit="papers", dynamic_ncols=True):
            if len(negatives) >= target_neg: break
            if not is_duplicate(ex["_paper"], "neg_C"): negatives.append(ex)
        log(f"    After [C]: {len(negatives):,}")

    rng = random.Random(seed)
    if len(negatives) > target_neg:
        rng.shuffle(negatives); negatives = negatives[:target_neg]
    if len(negatives) < target_neg:
        log(f"  WARNING: Only {len(negatives):,} negatives collected "
            f"(target={target_neg:,}). Consider lowering neg_ratio.")

    log(f"\n  ✓ Final negatives: {len(negatives):,}")

    # ── COMBINE + SPLIT ───────────────────────────────────────────────────────
    all_examples = positives + negatives
    rng.shuffle(all_examples)
    actual_ratio = len(negatives) / max(len(positives), 1)

    log(f"\n{'='*65}")
    log(f"DATASET SUMMARY")
    log(f"{'='*65}")
    log(f"  Positives : {len(positives):,}")
    log(f"  Negatives : {len(negatives):,}")
    log(f"  Total     : {len(all_examples):,}")
    log(f"  Ratio     : 1:{actual_ratio:.2f}")
    total_deduped = sum(dedup_stats.values())
    if total_deduped:
        log(f"  Dedup removed {total_deduped:,} total:")
        for k, v in sorted(dedup_stats.items()): log(f"    {k}: {v:,}")
    log(f"{'='*65}")

    splits = split_and_save(all_examples, cfg)

    # ── STATS FILE ────────────────────────────────────────────────────────────
    pos_src = defaultdict(int); neg_src = defaultdict(int)
    for e in positives: pos_src[e["source"]] += 1
    for e in negatives: neg_src[e["source"]] += 1

    stats = {
        "version":                    "v6",
        "total":                      len(all_examples),
        "positives":                  len(positives),
        "negatives":                  len(negatives),
        "actual_neg_ratio":           round(actual_ratio, 3),
        "config_min_score":           min_score,
        "config_hard_neg_max_score":  hard_neg_max_score,
        "config_neg_ratio":           neg_ratio,
        "excluded_score_range":       f"{hard_neg_max_score+1}–{min_score-1}",
        "num_workers":                num_workers,
        "raw_final_relevant":         len(final_rel),
        "score_distribution":         {str(k): v for k, v in sorted(score_dist.items())},
        "dedup_removed":              dict(dedup_stats),
        "dedup_total_removed":        total_deduped,
        "positive_sources":           dict(pos_src),
        "negative_sources":           dict(neg_src),
        "splits": {
            k: {"count": len(v),
                "pos":   sum(1 for r in v if r["label"] == 1),
                "neg":   sum(1 for r in v if r["label"] == 0)}
            for k, v in splits.items()
        },
        "input_fields_active": {
            k: v for k, v in cfg["input_fields"].items()
            if k.startswith("use_") and v
        },
    }
    stats_path = os.path.join(cfg["paths"]["dataset_dir"], "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    log(f"Stats saved -> {stats_path}")
    log("=" * 65)
    log("Done!")
    log(f"Run: python train_bert_classifier.py --config {args.config}")
    log("=" * 65)


if __name__ == "__main__":
    main()
