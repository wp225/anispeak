from __future__ import annotations

import sqlite3
from pathlib import Path
import pandas as pd

from anispeakarchive.core.config import load_settings
import lancedb


def main() -> None:
    s = load_settings("configs/default.yaml")
    db = lancedb.connect(s.paths.lancedb_dir)
    tbl = db.open_table(s.index.table_name)

    out_path = Path("cache/export.sqlite")
    if out_path.exists():
        out_path.unlink()

    con = sqlite3.connect(str(out_path))
    try:
        df = tbl.search().to_pandas()
        # vectors are big; drop to keep sqlite small unless you need it
        if "vector" in df.columns:
            df = df.drop(columns=["vector"])
        df.to_sql("papers", con, index=False)
        con.commit()
    finally:
        con.close()

    print(f"[OK] Exported to {out_path}")


if __name__ == "__main__":
    main()
