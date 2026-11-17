#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


DEFAULT_OUTPUT = Path("data/website.zip")
DEFAULT_INCLUDES = [Path("anispeakarchive")]

EXCLUDED_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "cache",
    "dist",
    "build",
}

EXCLUDED_FILE_NAMES = {
    ".DS_Store",
}

EXCLUDED_PATTERNS = [
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.log",
    "*.sqlite",
    "*.db",
    "*.gz",
    "*.zip",
    "*.parquet",
    "*.jsonl",
    "*.csv",
    "*.tsv",
    "*.pt",
    "*.bin",
    "*.safetensors",
]


def should_skip(path: Path) -> bool:
    parts = path.parts
    if any(part in EXCLUDED_DIR_NAMES for part in parts):
        return True
    if path.name in EXCLUDED_FILE_NAMES:
        return True
    if any(fnmatch.fnmatch(path.name, pattern) for pattern in EXCLUDED_PATTERNS):
        return True
    return False


def iter_files(root: Path, includes: list[Path]) -> list[Path]:
    files: list[Path] = []
    for include in includes:
        target = (root / include).resolve()
        if not target.exists():
            raise FileNotFoundError(f"Include path does not exist: {include}")

        if target.is_file():
            rel = target.relative_to(root.resolve())
            if not should_skip(rel):
                files.append(target)
            continue

        for dirpath, dirnames, filenames in os.walk(target):
            current_dir = Path(dirpath)
            dirnames[:] = [
                d for d in dirnames
                if d not in EXCLUDED_DIR_NAMES and not d.startswith(".")
            ]
            for name in filenames:
                candidate = current_dir / name
                rel = candidate.relative_to(root.resolve())
                if should_skip(rel):
                    continue
                files.append(candidate)
    return sorted(files)


def build_zip(root: Path, output: Path, includes: list[Path]) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    files = iter_files(root, includes)

    if output.exists():
        output.unlink()

    count = 0
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as zf:
        for file_path in files:
            arcname = file_path.relative_to(root)
            zf.write(file_path, arcname.as_posix())
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package source code into a zip for LLM context."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root to package from. Default: current working directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Zip output path relative to --root unless absolute. Default: data/website.zip",
    )
    parser.add_argument(
        "--include",
        type=Path,
        action="append",
        help="Path to include. Repeatable. Default: anispeakarchive",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else (root / args.output)
    includes = args.include if args.include else DEFAULT_INCLUDES

    count = build_zip(root, output.resolve(), includes)
    print(f"[OK] Wrote {count} files to {output.resolve()}")


if __name__ == "__main__":
    main()
