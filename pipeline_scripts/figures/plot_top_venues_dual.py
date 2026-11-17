#!/usr/bin/env python3
"""Top publication venues per research tradition — dual lollipop chart."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from _plot_utils import (
    LoadSpec,
    add_yaxis_status_markers,
    load_df,
    relative_rank_status_for_labels,
    value_counts_scalar,
)
from _plot_style import apply_style, save_both, PALETTE


def _topk(df, group: str, k: int):
    dfg = df[df["group2"] == group]
    assert len(dfg) > 0, f"No rows for group2={group}"
    vc = value_counts_scalar(dfg, "venue")
    assert len(vc) > 0, f"No non-empty venues for group2={group}"
    return vc.head(k).sort_values(ascending=True)


def _trend_statuses(df, group: str, labels: list[str]) -> dict[str, str]:
    focal = value_counts_scalar(df[df["group2"] == group], "venue")
    other_group = "comp_or_hybrid" if group == "biological" else "biological"
    other = value_counts_scalar(df[df["group2"] == other_group], "venue")
    return relative_rank_status_for_labels(
        focal.index.tolist(),
        other.index.tolist(),
        labels,
    )


def _lollipop(ax, series, title: str, subtitle: str, color: str, statuses: dict[str, str]) -> None:
    y_pos = range(len(series))
    # xmin=1 because log(0) is undefined; all venue counts are >= 1
    ax.hlines(list(y_pos), xmin=1, xmax=series.values, color=color,
              linewidth=1.4, alpha=0.7)
    ax.scatter(series.values, list(y_pos), color=color, s=42, zorder=3)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(series.index.tolist(), fontsize=8.0)
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.set_xlabel("Papers (log scale)")
    ax.set_xscale("log")
    ax.xaxis.grid(True, linewidth=0.4, alpha=0.45, linestyle="--", which="both")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", which="minor", labelsize=0)
    add_yaxis_status_markers(ax, statuses)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   required=True)
    ap.add_argument("--year_min", type=int, default=1950)
    ap.add_argument("--year_max", type=int, default=2026)
    ap.add_argument("--k",        type=int, default=15)
    ap.add_argument("--out_dir",  default="output/venues")
    args = ap.parse_args()

    apply_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(LoadSpec(config=args.config, year_min=args.year_min, year_max=args.year_max))
    assert "venue" in df.columns

    n_bio  = int((df["group2"] == "biological").sum())
    n_comp = int((df["group2"] == "comp_or_hybrid").sum())

    bio  = _topk(df, "biological",    args.k)
    comp = _topk(df, "comp_or_hybrid",args.k)

    bio_status  = _trend_statuses(df, "biological", bio.index.tolist())
    comp_status = _trend_statuses(df, "comp_or_hybrid", comp.index.tolist())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.8))

    _lollipop(ax1, bio,  "Top venues — Biological",
              f"(n = {n_bio:,} papers)",  PALETTE["biological"], bio_status)
    _lollipop(ax2, comp, "Top venues — Computational",
              f"(n = {n_comp:,} papers)", PALETTE["comp_or_hybrid"], comp_status)

    fig.suptitle("Most frequent publication venues (log scale).",
                 fontsize=11, fontweight="bold", y=0.975)
    fig.text(0.5, 0.01,
             "Markers compare rank against the other tradition: ↑ ranked higher here, ↓ ranked lower here, - same rank, * absent in the other tradition",
             ha="center", va="bottom", fontsize=7.1, color="#555555")
    fig.subplots_adjust(left=0.18, right=0.98, top=0.80, bottom=0.16, wspace=1.05)

    out_pdf = out_dir / "top_venues_dual.pdf"
    out_png = out_dir / "top_venues_dual.png"
    save_both(fig, out_pdf, out_png, tight=False)

    assert out_pdf.exists() and out_pdf.stat().st_size > 0
    assert out_png.exists() and out_png.stat().st_size > 0
    print(f"Wrote: {out_pdf}")
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
