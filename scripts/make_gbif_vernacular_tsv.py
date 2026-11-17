#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import os
import queue
import sys
import threading
import time
import zipfile
from dataclasses import dataclass
from typing import Optional, Dict, Any, Iterable, Tuple, List
csv.field_size_limit(sys.maxsize)
import requests


DEFAULT_BACKBONE_ZIP_URL = "https://hosted-datasets.gbif.org/datasets/backbone/current/backbone.zip"


def _assert(cond: bool, msg: str) -> None:
    assert cond, msg


def setup_logging(log_path: Optional[str], verbose: bool) -> logging.Logger:
    logger = logging.getLogger("gbif_vernacular")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def check_gbif_api(gbif_base: str) -> None:
    url = gbif_base.rstrip("/") + "/species/match"
    params = {"scientificName": "Panthera leo", "verbose": "false"}
    r = requests.get(url, params=params, timeout=10)
    _assert(r.status_code == 200, f"GBIF API check failed: {r.status_code} {r.text[:200]}")
    data = r.json()
    _assert(isinstance(data, dict), "GBIF API check: response is not a JSON object")
    _assert("usage" in data and isinstance(data["usage"], dict), "GBIF API check: missing usage")
    rank = data["usage"].get("rank")
    _assert(rank in {"SPECIES", "SUBSPECIES"}, f"GBIF API check: unexpected rank={rank!r}")


def download_file(url: str, out_path: str, logger: logging.Logger) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    logger.info(f"Downloading backbone zip -> {out_path}")
    with requests.get(url, stream=True, timeout=60) as r:
        _assert(r.status_code == 200, f"Download failed: {r.status_code} for {url}")
        total = 0
        t0 = time.time()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                total += len(chunk)
                if total and (time.time() - t0) > 1.0:
                    logger.info(f"downloaded_bytes={total}")
                    t0 = time.time()
    _assert(os.path.exists(out_path), f"Download did not create file: {out_path}")
    _assert(os.path.getsize(out_path) > 10_000_000, f"Downloaded zip too small: {os.path.getsize(out_path)} bytes")
    logger.info(f"Download done bytes={os.path.getsize(out_path)}")


def find_member_case_insensitive(zf: zipfile.ZipFile, target_basename: str) -> str:
    target_lower = target_basename.lower()
    candidates: List[str] = []
    for name in zf.namelist():
        base = name.split("/")[-1]
        if base.lower() == target_lower:
            candidates.append(name)
    _assert(len(candidates) >= 1, f"Could not find {target_basename} in zip. members={len(zf.namelist())}")
    candidates.sort(key=len)
    return candidates[0]


def parse_boolish(x: Optional[str]) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return "1"
    if s in {"false", "f", "0", "no", "n"}:
        return "0"
    return ""


def coerce_taxon_key(taxon_id: Optional[str]) -> Optional[int]:
    if taxon_id is None:
        return None
    s = str(taxon_id).strip()
    if s == "":
        return None
    if s.isdigit():
        return int(s)
    if "/" in s:
        tail = s.rsplit("/", 1)[-1]
        if tail.isdigit():
            return int(tail)
    return None


@dataclass
class Counters:
    rows_read: int = 0
    rows_kept: int = 0
    rows_written: int = 0
    rows_skipped: int = 0


class SafeCounters:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._c = Counters()

    def inc(self, field: str, n: int = 1) -> None:
        with self._lock:
            cur = getattr(self._c, field)
            setattr(self._c, field, cur + n)

    def snapshot(self) -> Counters:
        with self._lock:
            return Counters(**self._c.__dict__)


def worker_loop(
    in_q: "queue.Queue[Optional[Dict[str, Any]]]",
    out_q: "queue.Queue[Optional[Tuple[int, str, str, str]]]",
    counters: SafeCounters,
    require_language: bool,
) -> None:
    while True:
        item = in_q.get()
        if item is None:
            in_q.task_done()
            break

        counters.inc("rows_read", 1)

        taxon_key = coerce_taxon_key(item.get("taxonID") or item.get("taxonKey") or item.get("taxonId"))
        name = (item.get("vernacularName") or item.get("name") or "").strip()
        if taxon_key is None or name == "":
            counters.inc("rows_skipped", 1)
            in_q.task_done()
            continue

        lang = (item.get("language") or "").strip()
        if require_language and lang == "":
            counters.inc("rows_skipped", 1)
            in_q.task_done()
            continue

        pref = parse_boolish(item.get("isPreferredName"))

        counters.inc("rows_kept", 1)
        out_q.put((taxon_key, name, lang, pref))
        in_q.task_done()


def writer_loop(
    out_q: "queue.Queue[Optional[Tuple[int, str, str, str]]]",
    out_tsv: str,
    counters: SafeCounters,
    limit: Optional[int],
    logger: logging.Logger,
) -> None:
    os.makedirs(os.path.dirname(out_tsv) or ".", exist_ok=True)
    tmp_path = out_tsv + ".tmp"

    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["taxonKey", "vernacularName", "language", "isPreferredName"])

        wrote_any = False
        while True:
            item = out_q.get()
            if item is None:
                out_q.task_done()
                break

            if limit is not None:
                snap = counters.snapshot()
                if snap.rows_written >= limit:
                    out_q.task_done()
                    continue

            taxon_key, name, lang, pref = item
            w.writerow([taxon_key, name, lang, pref])
            wrote_any = True
            counters.inc("rows_written", 1)
            out_q.task_done()

        _assert(wrote_any, "Writer wrote zero rows. Input might be empty or filters too strict.")

    os.replace(tmp_path, out_tsv)
    _assert(os.path.exists(out_tsv), f"Output TSV not created: {out_tsv}")
    _assert(os.path.getsize(out_tsv) > 100, f"Output TSV too small, probably empty: {out_tsv}")
    logger.info(f"Writer finished out_tsv={out_tsv} bytes={os.path.getsize(out_tsv)}")


def progress_loop(counters: SafeCounters, every_s: float, logger: logging.Logger, stop_evt: threading.Event) -> None:
    t0 = time.time()
    last = Counters()
    last_t = t0

    while not stop_evt.wait(timeout=every_s):
        now = time.time()
        snap = counters.snapshot()

        dt = max(1e-6, now - last_t)
        d_read = snap.rows_read - last.rows_read
        d_written = snap.rows_written - last.rows_written
        r_read = d_read / dt
        r_written = d_written / dt

        elapsed = now - t0
        logger.info(
            "progress "
            f"elapsed_s={elapsed:.1f} "
            f"rows_read={snap.rows_read} rows_kept={snap.rows_kept} rows_written={snap.rows_written} rows_skipped={snap.rows_skipped} "
            f"read_rate={r_read:.1f}/s write_rate={r_written:.1f}/s"
        )

        last = snap
        last_t = now


def open_vernacular_tsv_from_zip(backbone_zip: str, logger: logging.Logger) -> Iterable[Dict[str, Any]]:
    _assert(zipfile.is_zipfile(backbone_zip), f"Not a valid zip: {backbone_zip}")
    zf = zipfile.ZipFile(backbone_zip, "r")
    member = find_member_case_insensitive(zf, "VernacularName.tsv")
    logger.info(f"Using member={member}")

    f = zf.open(member, "r")  # binary file-like
    text = io.TextIOWrapper(f, encoding="utf-8", errors="replace", newline="")
    reader = csv.DictReader(text, delimiter="\t")
    _assert(reader.fieldnames is not None and len(reader.fieldnames) > 2, "VernacularName.tsv header missing or malformed")
    _assert("vernacularName" in reader.fieldnames, f"Expected 'vernacularName' column, got {reader.fieldnames}")
    _assert(("taxonID" in reader.fieldnames) or ("taxonKey" in reader.fieldnames), f"Expected taxonID/taxonKey column, got {reader.fieldnames}")
    return reader  # NOTE: reader depends on zf staying alive; keep zf referenced


def main() -> None:
    import io  # local import so script stays single-file

    ap = argparse.ArgumentParser(description="Extract GBIF backbone vernacular names into a compact TSV with progress + threading.")
    ap.add_argument("--gbif-base", default="http://127.0.0.1:5643/v2")
    ap.add_argument("--backbone-zip-url", default=DEFAULT_BACKBONE_ZIP_URL)
    ap.add_argument("--backbone-zip", default="data/gbif_backbone.zip")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--out-tsv", default="data/gbif_vernacular.tsv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--require-language", action="store_true")

    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--queue-size", type=int, default=20000)
    ap.add_argument("--progress-every", type=float, default=2.0)
    ap.add_argument("--log", default=None)
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    _assert(args.workers >= 1, "--workers must be >= 1")
    _assert(args.queue_size >= 1000, "--queue-size must be >= 1000")
    _assert(args.progress_every >= 0.2, "--progress-every too small")

    logger = setup_logging(args.log, args.verbose)

    logger.info("Starting")
    check_gbif_api(args.gbif_base)
    logger.info("GBIF API OK")

    if args.no_download:
        _assert(os.path.exists(args.backbone_zip), f"--no-download set but zip not found: {args.backbone_zip}")
    else:
        if not os.path.exists(args.backbone_zip):
            download_file(args.backbone_zip_url, args.backbone_zip, logger)
        else:
            logger.info(f"Zip exists, skipping download: {args.backbone_zip}")

    _assert(os.path.exists(args.backbone_zip), f"Zip not found after download/skip: {args.backbone_zip}")

    counters = SafeCounters()
    in_q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=args.queue_size)
    out_q: "queue.Queue[Optional[Tuple[int, str, str, str]]]" = queue.Queue(maxsize=args.queue_size)

    stop_evt = threading.Event()
    prog = threading.Thread(
        target=progress_loop,
        args=(counters, args.progress_every, logger, stop_evt),
        daemon=True,
    )
    prog.start()

    writer = threading.Thread(
        target=writer_loop,
        args=(out_q, args.out_tsv, counters, args.limit, logger),
        daemon=True,
    )
    writer.start()

    workers: List[threading.Thread] = []
    for _ in range(args.workers):
        t = threading.Thread(
            target=worker_loop,
            args=(in_q, out_q, counters, args.require_language),
            daemon=True,
        )
        t.start()
        workers.append(t)

    # Feed rows
    t_feed0 = time.time()
    fed = 0
    logger.info("Reading VernacularName.tsv and feeding workers")

    # Keep zip file alive by holding refs
    _assert(zipfile.is_zipfile(args.backbone_zip), f"Not a valid zip: {args.backbone_zip}")
    with zipfile.ZipFile(args.backbone_zip, "r") as zf:
        member = find_member_case_insensitive(zf, "VernacularName.tsv")
        logger.info(f"Using member={member}")
        with zf.open(member, "r") as bf:
            import io as _io
            text = _io.TextIOWrapper(bf, encoding="utf-8", errors="replace", newline="")
            reader = csv.DictReader(text, delimiter="\t")
            _assert(reader.fieldnames is not None and len(reader.fieldnames) > 2, "VernacularName.tsv header missing/malformed")
            _assert("vernacularName" in reader.fieldnames, f"Missing vernacularName col: {reader.fieldnames}")
            _assert(("taxonID" in reader.fieldnames) or ("taxonKey" in reader.fieldnames), f"Missing taxonID/taxonKey col: {reader.fieldnames}")

            for row in reader:
                # Optional early stop based on written limit is handled by writer,
                # but we can also stop feeding if we’re way past it.
                if args.limit is not None and counters.snapshot().rows_written >= args.limit:
                    break

                in_q.put(row)
                fed += 1
                if fed % 500000 == 0:
                    dt = max(1e-6, time.time() - t_feed0)
                    logger.info(f"fed_rows={fed} feed_rate={fed/dt:.1f}/s")

    logger.info("Feed complete. Signaling workers to stop.")
    for _ in workers:
        in_q.put(None)

    in_q.join()
    logger.info("All worker input consumed. Signaling writer to stop.")
    out_q.put(None)
    out_q.join()

    stop_evt.set()
    writer.join(timeout=5)

    final = counters.snapshot()
    logger.info(
        "Done "
        f"rows_read={final.rows_read} rows_kept={final.rows_kept} rows_written={final.rows_written} rows_skipped={final.rows_skipped}"
    )

    _assert(final.rows_written > 0, "No rows were written. Something is wrong.")
    _assert(os.path.exists(args.out_tsv), f"Output TSV missing: {args.out_tsv}")

    # Extra sanity: file should be growing and not empty
    size = os.path.getsize(args.out_tsv)
    _assert(size > 1024, f"Output TSV too small: {size} bytes")
    logger.info(f"Output OK out_tsv={args.out_tsv} bytes={size}")


if __name__ == "__main__":
    main()
