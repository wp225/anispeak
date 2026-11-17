#!/usr/bin/env python3
"""
Publication trends as a stacked area plot.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _plot_utils import LoadSpec, load_df
from _plot_style import apply_style, save_both, PALETTE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   required=True)
    ap.add_argument("--year_min", type=int, default=1950)
    ap.add_argument("--year_max", type=int, default=2026)
    ap.add_argument("--out_dir",  default="output/trends")
    args = ap.parse_args()

    apply_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(LoadSpec(config=args.config, year_min=args.year_min, year_max=args.year_max))
    df = df.copy()

    df["year_bucket"] = df["year"].astype(int)

    all_years = list(range(args.year_min, args.year_max + 1))
    idx = pd.Index(all_years, dtype=int)
    counts = (
        df.groupby(["year_bucket", "group2"]).size()
          .unstack(fill_value=0)
          .reindex(index=idx, fill_value=0)
    )
    assert "biological"    in counts.columns, "Missing 'biological' column"
    assert "comp_or_hybrid" in counts.columns, "Missing 'comp_or_hybrid' column"

    x        = np.array(counts.index, dtype=int)
    y_bio    = counts["biological"].to_numpy(dtype=float)
    y_comp   = counts["comp_or_hybrid"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 3.2))

    col_bio  = PALETTE["biological"]
    col_comp = PALETTE["comp_or_hybrid"]

    ax.stackplot(
        x,
        y_bio,
        y_comp,
        colors=[col_bio, col_comp],
        alpha=0.8,
        labels=["Biological", "Computational"],
    )
    ax.plot(x, y_bio, color=col_bio, linewidth=1.0)
    ax.plot(x, y_bio + y_comp, color=col_comp, linewidth=1.0)

    ax.yaxis.grid(True, linewidth=0.4, alpha=0.45, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylabel("Papers per year")
    ax.set_xlabel("Year")

    ticks = list(range(args.year_min, args.year_max + 1, 5))
    if args.year_max not in ticks:
        ticks.append(args.year_max)
    ticks = sorted(set(ticks))
    if len(ticks) >= 2 and ticks[-1] - ticks[-2] <= 1:
        ticks.pop(-2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(args.year_min, args.year_max)

    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=8.5)
    ax.set_title("Publication trends in animal communication research",
                 fontsize=11, fontweight="bold")

    out_pdf = out_dir / "publication_trends_two_groups.pdf"
    out_png = out_dir / "publication_trends_two_groups.png"
    save_both(fig, out_pdf, out_png, tight=True)

    assert out_pdf.exists() and out_pdf.stat().st_size > 0
    assert out_png.exists() and out_png.stat().st_size > 0
    print(f"Wrote: {out_pdf}")
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
