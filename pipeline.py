#!/usr/bin/env python3
"""
pipeline.py — AniSpeakArchive end-to-end update pipeline

Stages (run in order):
  1. download  — pull new OpenAlex snapshot partitions from S3
  2. bert      — BERT relevance classification on new partitions
  3. qwen      — Qwen2.5-14B metadata extraction and annotation
  4. index     — rebuild LanceDB search index for the web app

Each stage is resumable: already-downloaded files, already-classified papers,
and already-indexed docs are all skipped automatically.

Usage:
  python pipeline.py                      # run all 4 stages once
  python pipeline.py --stage bert         # run one stage only (e.g. for recovery)
  python pipeline.py --schedule 7d        # repeat full pipeline every 7 days
  python pipeline.py --schedule 12h --stage download   # schedule single stage
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Missing dependency: pip install pyyaml")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / "pipeline_state.json"
LOG_FILE = ROOT / "pipeline.log"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with LOG_FILE.open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Config / state
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    p = ROOT / path
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open() as f:
        return yaml.safe_load(f) or {}


def parse_interval(s: str) -> int:
    """Parse '7d', '12h', '30m' into seconds."""
    m = re.fullmatch(r"(\d+)([dhm])", s.strip().lower())
    if not m:
        raise ValueError(f"Invalid schedule interval '{s}'. Examples: 7d, 12h, 30m")
    n, unit = int(m.group(1)), m.group(2)
    return n * {"d": 86400, "h": 3600, "m": 60}[unit]


def load_state() -> dict:
    if STATE_FILE.exists():
        with STATE_FILE.open() as f:
            return json.load(f)
    return {}


def save_state(state: dict) -> None:
    with STATE_FILE.open("w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------

def run(cmd: list[str], cwd: Path, extra_env: dict[str, str] | None = None) -> None:
    log(f"$ {' '.join(cmd)}")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"Stage failed (exit {result.returncode}): {' '.join(cmd)}"
        )


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_download(cfg: dict, test: bool = False) -> None:
    log("=== Stage 1: download ===")
    paths = cfg["paths"]
    dl_cfg = cfg.get("download", {})

    snapshot_dir = ROOT / paths["snapshot_dir"]
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "pipeline_scripts" / "download_snapshot.py"),
        "--output_dir", str(snapshot_dir),
        "--workers",    str(dl_cfg.get("workers", 8)),
    ]
    if test:
        cmd.append("--test")
    run(cmd, cwd=ROOT / "pipeline_scripts")


def stage_bert(cfg: dict) -> None:
    log("=== Stage 2: BERT filter ===")
    paths   = cfg["paths"]
    bert_cfg = cfg.get("bert", {})

    snapshot_dir = ROOT / paths["snapshot_dir"]
    output_dir   = ROOT / paths["bert_output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    model  = ROOT / paths["bert_model"]
    config = ROOT / paths["bert_config"]
    year_end = str(bert_cfg.get("year_end", datetime.now().year))

    run(
        [
            sys.executable, "-u",
            str(ROOT / "pipeline_scripts" / "filter_bert.py"),
            "--snapshot_dir", str(snapshot_dir),
            "--output_dir",   str(output_dir),
            "--model",        str(model),
            "--config",       str(config),
            "--gpu_configs",  bert_cfg.get("gpu_configs", "0:batch=2048,chunk=16384,queue=4,readers=2"),
            "--threshold",    str(bert_cfg.get("threshold", 0.45)),
            "--year_start",   str(bert_cfg.get("year_start", 2000)),
            "--year_end",     year_end,
            "--seed",         str(bert_cfg.get("seed", 42)),
            "--resume",
        ],
        cwd=ROOT / "pipeline_scripts",
    )


def stage_qwen(cfg: dict) -> None:
    log("=== Stage 3: Qwen annotation ===")
    paths    = cfg["paths"]
    qwen_cfg = cfg.get("qwen", {})

    output_dir = ROOT / paths["bert_output_dir"]
    qwen_model = paths.get("qwen_model", "/supernova/data/lekhak/models/Qwen2.5-14B-Instruct")

    run(
        [
            sys.executable,
            str(ROOT / "pipeline_scripts" / "annotate_qwen.py"),
            "--output_dir",   str(output_dir),
            "--model",        qwen_model,
            "--gpu_mem_util", str(qwen_cfg.get("gpu_mem_util", 0.80)),
            "--reprocess",
        ],
        cwd=ROOT / "pipeline_scripts",
    )


def stage_index(cfg: dict) -> None:
    log("=== Stage 4: rebuild index ===")
    paths = cfg["paths"]

    final_jsonl = ROOT / paths["bert_output_dir"] / "final_relevant.jsonl"
    if not final_jsonl.exists():
        raise FileNotFoundError(
            f"final_relevant.jsonl not found at {final_jsonl}. "
            "Run stages 2 and 3 first."
        )

    run(
        [sys.executable, "-m", "scripts.build_index"],
        cwd=ROOT,
        extra_env={"PIPELINE_FINAL_JSONL": str(final_jsonl)},
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

STAGES: dict[str, object] = {
    "download": stage_download,
    "bert":     stage_bert,
    "qwen":     stage_qwen,
    "index":    stage_index,
}


def run_pipeline(cfg: dict, stage: str | None = None, test: bool = False) -> None:
    t0 = time.time()
    names = [stage] if stage else list(STAGES.keys())

    for name in names:
        log(f"--- starting stage: {name} ---")
        try:
            if name == "download":
                stage_download(cfg, test=test)
            else:
                STAGES[name](cfg)  # type: ignore[operator]
        except Exception as e:
            log(f"ERROR in stage '{name}': {e}")
            raise

    elapsed = int(time.time() - t0)
    log(f"Pipeline finished in {elapsed // 60}m{elapsed % 60:02d}s")

    state = load_state()
    state["last_run"] = datetime.now().isoformat()
    state["last_stage"] = stage or "all"
    save_state(state)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AniSpeakArchive end-to-end pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage", choices=list(STAGES.keys()), default=None,
        help="Run a single stage (default: all stages in order)",
    )
    parser.add_argument(
        "--schedule", default=None, metavar="INTERVAL",
        help="Repeat on a fixed interval, e.g. 7d, 12h, 30m",
    )
    parser.add_argument(
        "--config", default="pipeline_config.yaml",
        help="Pipeline config file (default: pipeline_config.yaml)",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: download only 3 files (smoke-test the pipeline)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.schedule:
        interval = parse_interval(args.schedule)
        log(f"Scheduler started — running every {args.schedule}")
        while True:
            try:
                run_pipeline(cfg, stage=args.stage, test=args.test)
            except Exception as e:
                log(f"Pipeline run failed: {e}. Will retry after {args.schedule}.")
            log(f"Sleeping {args.schedule} until next run...")
            time.sleep(interval)
    else:
        run_pipeline(cfg, stage=args.stage, test=args.test)


if __name__ == "__main__":
    main()
