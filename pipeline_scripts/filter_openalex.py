#!/usr/bin/env python3
"""
filter_snapshot.py
==================
Multi-process snapshot filter with:
  - Two-thread pipeline per worker (I/O thread + parse/filter thread overlap)
  - Per-worker private shard files  → zero inter-worker write contention
  - Merge step in main process after all workers finish
  - tqdm live bars (per-file progress + overall)
  - Writes ONLY the fields fetch_papers.py actually reads
  - Largest files dispatched first (no stragglers)
  - Resumable: skips files already written via a .done sidecar
  - Bounded queue (IO_QSIZE=64) → backpressure prevents RAM exhaustion

Usage:
    python filter_snapshot.py \
        --snapshot_dir /supernova/data/lekhak/temp/animal_papers/build_corpus_v3/openalex-works \
        --out_jsonl    ./filtered_works.jsonl \
        --workers      10 \
        --year_start   2000 \
        --year_end     2026
"""

import argparse
import gzip
import multiprocessing as mp
import os
import queue
import re
import shutil
import threading
import time
from datetime import datetime
from glob import glob

try:
    import orjson
    def jloads(b): return orjson.loads(b)
    def jdumps(o): return orjson.dumps(o).decode("utf-8")
except ImportError:
    import json
    def jloads(b): return json.loads(b.decode("utf-8") if isinstance(b, bytes) else b)
    def jdumps(o): return json.dumps(o, ensure_ascii=False, separators=(",", ":"))

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# ONLY THE FIELDS fetch_papers.py ACTUALLY READS
# ============================================================
KEEP_FIELDS = frozenset({
    "id", "doi", "display_name", "title",
    "publication_year", "publication_date",
    "type", "language",
    "primary_location", "locations", "best_oa_location",
    "open_access",
    "authorships",
    "topics", "concepts", "keywords",
    "abstract_inverted_index",
    "biblio",
    "cited_by_count", "counts_by_year",
    "is_retracted", "is_paratext",
    "referenced_works", "related_works",
    "grants",
    "sustainable_development_goals",
    "mesh",
    "primary_topic",
    "indexed_in",
    "updated_date", "created_date",
})


def slim(raw: dict) -> dict:
    """Keep only fields needed by fetch_papers.py."""
    return {k: v for k, v in raw.items() if k in KEEP_FIELDS}


# ============================================================
# BLACKLIST / ACCEPT
# ============================================================
HARD_BLACKLIST = [
    "clinical trial", "patient cohort", "chemotherapy", "radiotherapy",
    "oncology", "metastasis", "carcinoma", "tumor staging",
    "surgical procedure", "anesthesia protocol", "icu patient",
    "covid-19 pandemic", "sars-cov-2",
    "fault diagnosis", "bearing fault", "gearbox vibration", "turbine blade",
    "predictive maintenance", "structural health monitoring",
    "stock market", "cryptocurrency", "portfolio optimization", "trading strategy",
    "music recommendation", "playlist generation", "music genre classification",
    "whale optimization algorithm", "grey wolf optimizer", "bat algorithm",
    "firefly algorithm", "ant colony optimization", "particle swarm optimization",
    "grasshopper optimization", "moth-flame optimization", "artificial bee colony",
    "dragonfly algorithm", "salp swarm algorithm", "marine predator algorithm",
    "harris hawk optimization", "golden eagle optimizer", "cuckoo search algorithm",
    "automatic speech recognition", "text-to-speech synthesis",
    "voice conversion system", "speaker verification", "speaker diarization",
    "speech emotion recognition human", "human speech enhancement",
]

STRONG_ACCEPT_KEYWORDS = [
    "bioacoustics", "animal vocalization", "bird song", "birdsong",
    "whale song", "dolphin whistle", "bat echolocation", "frog call",
    "insect sound", "primate call", "animal communication",
    "passive acoustic monitoring", "soundscape ecology",
    "vocal repertoire", "call type", "acoustic species identification",
    "vocal learning", "dawn chorus", "ultrasonic vocalization",
    "cetacean acoustic", "avian vocalization", "anuran call", "stridulation",
    "birdnet", "birdclef", "animal sound dataset", "earth species",
    "dog bark", "cat meow", "animal welfare vocalization",
    "interspecies communication", "animal translation",
    "bioacoustic classification", "ecoacoustics",
    "animal audio", "wildlife sound", "animal sound event",
    "elephant rumble", "wolf howl", "parrot speech", "fish sound",
    "marine mammal acoustic", "hydrophone recording",
        # Species-specific vocalizations
    "canine vocalization", "feline vocalization", "bovine vocalization",
    "equine vocalization", "rodent vocalization", "avian call",
    "songbird", "oscine", "suboscine", "gibbon call", "orangutan call",
    "chimpanzee vocalization", "corvid", "owl call", "raptor vocalization",
    "amphibian call", "cricket stridulation", "cicada song",
    "beehive sound", "insect bioacoustics", "reptile vocalization",
    "turtle vocalization", "crocodilian vocalization",

    # Dataset / benchmark papers
    "bioacoustic dataset", "animal sound corpus", "wildlife recording",
    "annotated animal", "acoustic benchmark", "audio dataset animal",
    "species classification dataset", "bird audio detection",
    "xeno-canto", "macaulay library", "freefield1010",
    "warblr", "dcase bioacoustic", "aesdd", "esc-50 animal",
    "rfcx rainforest", "google audioset animal", "dogspeak", "beans"

    # ML/DL methods specific to animal audio
    "animal sound classification", "species identification deep learning",
    "acoustic species detection", "bird species classification",
    "call segmentation", "vocalization segmentation",
    "automated bird recognition", "bioacoustic deep learning",
    "convolutional animal", "spectrogram animal",
]

WANTED_TOPIC_IDS = {
    "T11665", "T11228", "T10659", "T11349", "T10118",
    "T12130", "T10934", "T11560", "T11048", "T10808",
    "T11738", "T10289", "T12210", "T14217", "T11944",
    "T11387", "T12477", "T12891", "T12654", "T12003",
    "T13269", "T10651", "T11827", "T13080", "T13661",
    "T10891", "T11203", "T10443", "T13901", "T10042",
}

# Set once per worker process in _init_worker — never shared between processes
_BLACKLIST_RE    = None
_TOPIC_RE        = re.compile(r'T(\d+)')
_WORKER_OUT_PATH = None   # each process writes to its own private shard file


def _init_worker(shard_dir: str):
    """
    Called once per worker process at pool startup.
      - Compiles blacklist regexes (shared across all tasks in this process)
      - Assigns a unique shard output path keyed to the OS PID
        so no two workers ever write to the same file
    """
    global _BLACKLIST_RE, _WORKER_OUT_PATH
    _BLACKLIST_RE = [
        re.compile(r'\b' + re.escape(t) + r'\b', re.IGNORECASE)
        for t in HARD_BLACKLIST
    ]
    pid = os.getpid()
    _WORKER_OUT_PATH = os.path.join(shard_dir, f"shard_{pid}.jsonl")


# ============================================================
# FILTER HELPERS
# ============================================================

def _topic_ids(raw: dict) -> set:
    ids = set()
    for t in (raw.get("topics") or []):
        m = _TOPIC_RE.search(str(t.get("id") or ""))
        if m:
            ids.add("T" + m.group(1))
    return ids


def _filter_text(raw: dict) -> str:
    title = (raw.get("display_name") or raw.get("title") or "").lower()

    abstract = ""
    inv = raw.get("abstract_inverted_index")
    if inv:
        try:
            pairs = [(pos, w) for w, positions in inv.items() for pos in positions]
            pairs.sort()
            abstract = " ".join(w for _, w in pairs).lower()
        except Exception:
            pass

    topics   = " ".join(
        t.get("display_name", "") for t in (raw.get("topics")   or [])
    ).lower()
    concepts = " ".join(
        c.get("display_name", "") for c in (raw.get("concepts") or [])
    ).lower()
    keywords = " ".join(
        k.get("display_name", "") if isinstance(k, dict) else str(k)
        for k in (raw.get("keywords") or [])
    ).lower()
    return f"{title} {abstract} {topics} {concepts} {keywords}"


def matches(raw: dict, year_start: int, year_end: int) -> bool:
    y = raw.get("publication_year")
    if isinstance(y, int) and not (year_start <= y <= year_end):
        return False
    if raw.get("is_retracted") or raw.get("is_paratext"):
        return False
    lang = (raw.get("language") or "").strip().lower()
    if lang and lang != "en":
        return False

    text = _filter_text(raw)

    for pat in _BLACKLIST_RE:
        if pat.search(text):
            return False

    # Positive gate: topic ID match OR ≥2 strong-accept keywords
    if _topic_ids(raw) & WANTED_TOPIC_IDS:
        return True
    if sum(1 for kw in STRONG_ACCEPT_KEYWORDS if kw in text) >= 2:
        return True

    return False


# ============================================================
# TWO-THREAD WORKER  (I/O thread → parse/filter thread)
# ============================================================
_SENTINEL = object()
IO_QSIZE  = 64   # bounded → I/O thread blocks when consumer is slow (backpressure)


def _io_thread(gz_path: str, line_queue: queue.Queue) -> None:
    """Thread A: stream raw gzip lines into the bounded queue."""
    try:
        with gzip.open(gz_path, "rb") as f:
            for line in f:
                line_queue.put(line)   # blocks when queue is full → natural throttle
    except EOFError:
        pass   # truncated file — process whatever was read
    except Exception as e:
        line_queue.put(("__error__", str(e)))
    finally:
        line_queue.put(_SENTINEL)


def process_one_file(args: tuple) -> tuple:
    """
    Runs inside a pool worker process (Thread B).
    Consumes the line queue, parses/filters each record, and appends
    matching records to this process's private shard file.
    No other worker ever touches this file → zero file contention.

    Returns: (gz_path, checked, kept, error_str_or_None)
    """
    gz_path, year_start, year_end, done_marker_dir = args

    checked  = 0
    kept     = 0
    error    = None
    out_path = _WORKER_OUT_PATH   # unique per process, set in _init_worker

    line_queue: queue.Queue = queue.Queue(maxsize=IO_QSIZE)
    io = threading.Thread(
        target=_io_thread, args=(gz_path, line_queue), daemon=True
    )
    io.start()

    try:
        # 2 MB write buffer: efficient I/O without holding large OS buffers per worker
        with open(out_path, "a", encoding="utf-8", buffering=2 * 1024 * 1024) as out:
            while True:
                item = line_queue.get()

                if item is _SENTINEL:
                    break

                if isinstance(item, tuple) and item[0] == "__error__":
                    error = item[1]
                    break

                line = item.strip()
                if not line:
                    del item
                    continue

                try:
                    raw = jloads(line)
                except Exception:
                    del line, item
                    continue
                finally:
                    del item   # drop raw gzip bytes immediately after parse

                checked += 1

                if matches(raw, year_start, year_end):
                    record = jdumps(slim(raw))
                    out.write(record + "\n")
                    kept += 1
                    del record

                del raw    # CPython refcount → freed immediately, no GC wait
                del line

                if checked % 50_000 == 0:
                    out.flush()   # periodically release userspace write buffer

    except Exception as e:
        error = str(e)
    finally:
        io.join(timeout=5)

    return (gz_path, checked, kept, error)


# ============================================================
# MERGE: stream-concatenate all per-worker shards → final output
# ============================================================

def merge_shards(shard_dir: str, out_jsonl: str) -> None:
    """
    Sequentially concatenate every shard_*.jsonl in shard_dir into
    out_jsonl using 16 MB copy chunks.  Pure I/O — no parsing.
    """
    shards = sorted(glob(os.path.join(shard_dir, "shard_*.jsonl")))
    if not shards:
        log("WARNING: no shard files found to merge.")
        return

    log(f"Merging {len(shards)} shard(s) → {out_jsonl}")
    with open(out_jsonl, "wb") as fout:
        for shard in shards:
            size_mb = os.path.getsize(shard) / 1e6
            with open(shard, "rb") as fin:
                shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)
            log(f"  + {os.path.basename(shard)}  ({size_mb:.1f} MB)")


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot_dir", required=True)
    ap.add_argument("--out_jsonl",    required=True)
    ap.add_argument("--workers",      type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--pattern",      default="**/*.gz")
    ap.add_argument("--year_start",   type=int, default=2000)
    ap.add_argument("--year_end",     type=int, default=2026)
    ap.add_argument("--resume",       action="store_true",
                    help="Skip .gz files that already have a .done sidecar")
    ap.add_argument("--keep_shards",  action="store_true",
                    help="Retain per-worker shard files after merging")
    args = ap.parse_args()

    # ── locate input files ────────────────────────────────────────────────────
    gz_files = sorted(
        glob(os.path.join(args.snapshot_dir, args.pattern), recursive=True),
        key=os.path.getsize, reverse=True   # largest first → no stragglers
    )
    gz_files = [f for f in gz_files if not f.endswith(".part")]

    if not gz_files:
        log(f"No .gz files found under {args.snapshot_dir}")
        return

    # ── prepare directories ───────────────────────────────────────────────────
    done_marker_dir = args.out_jsonl + ".done_markers"
    shard_dir       = args.out_jsonl + ".shards"
    os.makedirs(done_marker_dir, exist_ok=True)
    os.makedirs(shard_dir,       exist_ok=True)

    # ── resume / fresh-run logic ──────────────────────────────────────────────
    if args.resume:
        todo    = []
        skipped = 0
        for f in gz_files:
            marker = os.path.join(done_marker_dir, os.path.basename(f) + ".done")
            if os.path.exists(marker):
                skipped += 1
            else:
                todo.append(f)
        log(f"Resume mode: skipping {skipped:,} already-done files")
        gz_files = todo
    else:
        # Fresh run — wipe previous output and all stale shards
        if os.path.exists(args.out_jsonl):
            log(f"Removing existing {args.out_jsonl}")
            os.remove(args.out_jsonl)
        for stale in glob(os.path.join(shard_dir, "shard_*.jsonl")):
            os.remove(stale)

    if not gz_files:
        log("All files already done. Nothing to do.")
        return

    log(f"Files to process : {len(gz_files):,}")
    log(f"Workers          : {args.workers}")
    log(f"Year range       : {args.year_start}–{args.year_end}")
    log(f"Shard dir        : {shard_dir}")
    log(f"Output           : {args.out_jsonl}")

    # out_path is NOT in worker_args — each worker derives its own path from
    # _WORKER_OUT_PATH (set in _init_worker via PID) → no shared file state
    worker_args = [
        (p, args.year_start, args.year_end, done_marker_dir)
        for p in gz_files
    ]

    t0            = time.time()
    total_checked = 0
    total_kept    = 0
    errors        = 0

    bar_kw = dict(
        total=len(gz_files),
        unit="file",
        desc="Filtering",
        dynamic_ncols=True,
        smoothing=0.1,
    )

    with mp.Pool(
        args.workers,
        initializer=_init_worker,
        initargs=(shard_dir,)
    ) as pool:

        results_iter = pool.imap_unordered(process_one_file, worker_args)

        if HAS_TQDM:
            with tqdm(**bar_kw) as pbar:
                for gz_path, checked, kept, err in results_iter:
                    total_checked += checked
                    total_kept    += kept
                    if err:
                        errors += 1
                        pbar.write(f"ERROR {os.path.basename(gz_path)}: {err}")
                    else:
                        marker = os.path.join(done_marker_dir,
                                              os.path.basename(gz_path) + ".done")
                        open(marker, "w").close()

                    elapsed = time.time() - t0
                    rate    = total_checked / max(elapsed, 1)
                    pbar.set_postfix({
                        "checked": f"{total_checked/1e6:.1f}M",
                        "kept":    f"{total_kept/1e3:.1f}k",
                        "rec/s":   f"{rate:,.0f}",
                        "err":     errors,
                    }, refresh=False)
                    pbar.update(1)
        else:
            done_n = 0
            for gz_path, checked, kept, err in results_iter:
                total_checked += checked
                total_kept    += kept
                done_n        += 1
                if err:
                    errors += 1
                    log(f"ERROR {os.path.basename(gz_path)}: {err}")
                else:
                    marker = os.path.join(done_marker_dir,
                                          os.path.basename(gz_path) + ".done")
                    open(marker, "w").close()
                elapsed = time.time() - t0
                rate    = total_checked / max(elapsed, 1)
                pct     = 100 * done_n / len(gz_files)
                log(f"  [{done_n:>5}/{len(gz_files)}] {pct:4.1f}%  "
                    f"checked={total_checked/1e6:.2f}M  "
                    f"kept={total_kept:,}  "
                    f"rate={rate:,.0f} rec/s  "
                    f"err={errors}")

    # ── merge all shards into the single final output file ────────────────────
    merge_shards(shard_dir, args.out_jsonl)

    if not args.keep_shards:
        for shard in glob(os.path.join(shard_dir, "shard_*.jsonl")):
            os.remove(shard)
        log("Shard files removed  (pass --keep_shards to retain them)")

    # ── final summary ─────────────────────────────────────────────────────────
    dt      = time.time() - t0
    rate    = total_checked / max(dt, 1)
    size_mb = os.path.getsize(args.out_jsonl) / 1e6 if os.path.exists(args.out_jsonl) else 0

    log("=" * 60)
    log("FILTER COMPLETE")
    log(f"  Files processed  : {len(gz_files):,}")
    log(f"  Records checked  : {total_checked:,}")
    log(f"  Records kept     : {total_kept:,}  "
        f"({100*total_kept/max(total_checked,1):.3f}%)")
    log(f"  Errors           : {errors:,}")
    log(f"  Time             : {dt/3600:.2f} h")
    log(f"  Throughput       : {rate:,.0f} rec/s")
    log(f"  Output size      : {size_mb:,.1f} MB")
    log(f"  Output file      : {args.out_jsonl}")
    log("=" * 60)


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
