#!/usr/bin/env python3
"""
Top species per research tradition — dual horizontal bar chart.

Species labels are normalized via _plot_utils.norm_species_label:
  - Authority/year suffixes stripped  (e.g. "Taeniopygia guttata (Vieillot, 1817)")
  - Synonym canonicalization          (e.g. "Poephila guttata" → "Zebra finch")
  - Common-name singularization       (e.g. "zebra finches" → "zebra finch")
  - Null/unknown values mapped to "Other" and excluded from ranking
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from _plot_utils import (
    LoadSpec,
    add_yaxis_status_markers,
    load_df,
    norm_species_label,
    relative_rank_status_for_labels,
    value_counts_list,
)
from _plot_style import apply_style, save_both, PALETTE


def _counts_for_group(df, col: str):
    return value_counts_list(
        df,
        col,
        normalizer=norm_species_label,
        drop_labels={"Other", "Others", "other species"},
    )


def _trend_statuses(df, labels: list[str], group: str) -> dict[str, str]:
    focal = _counts_for_group(df[df["group2"] == group].copy(), "species")
    other_group = "comp_or_hybrid" if group == "biological" else "biological"
    other = _counts_for_group(df[df["group2"] == other_group].copy(), "species")
    return relative_rank_status_for_labels(
        focal.index.tolist(),
        other.index.tolist(),
        labels,
    )


def _plot_panel(ax, vc, title: str, subtitle: str, top_n: int, color: str, statuses: dict[str, str]) -> None:
    vc = vc.head(top_n).sort_values(ascending=True)
    y_pos = range(len(vc))

    # Horizontal lollipop: line + dot, cleaner than solid bars
    ax.hlines(y=list(y_pos), xmin=0, xmax=vc.values, color=color, linewidth=1.4, alpha=0.7)
    ax.scatter(vc.values, list(y_pos), color=color, s=40, zorder=3)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(vc.index.tolist(), fontsize=8.2)
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.set_xlabel("Papers")
    ax.xaxis.grid(True, linewidth=0.4, alpha=0.45, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    add_yaxis_status_markers(ax, statuses)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--year_min", type=int, default=1950)
    ap.add_argument("--year_max", type=int, default=2026)
    ap.add_argument("--top_n",   type=int, default=15)
    ap.add_argument("--out_dir", default="output/species")
    args = ap.parse_args()
    assert args.top_n >= 5

    apply_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(LoadSpec(config=args.config, year_min=args.year_min, year_max=args.year_max))
    assert "group2" in df.columns

    df_trad = df[df["group2"] == "biological"].copy()
    df_comp = df[df["group2"] == "comp_or_hybrid"].copy()
    assert len(df_trad) > 0 and len(df_comp) > 0

    vc_trad = _counts_for_group(df_trad, "species")
    vc_comp = _counts_for_group(df_comp, "species")
    assert len(vc_trad) > 0 and len(vc_comp) > 0

    n_trad = len(df_trad)
    n_comp = len(df_comp)

    trad_status = _trend_statuses(df, vc_trad.head(args.top_n).index.tolist(), "biological")
    comp_status = _trend_statuses(df, vc_comp.head(args.top_n).index.tolist(), "comp_or_hybrid")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=False)

    _plot_panel(ax1, vc_trad,
                "Top species — Biological",
                f"(n = {n_trad:,} papers)",
                args.top_n, PALETTE["biological"], trad_status)

    _plot_panel(ax2, vc_comp,
                "Top species — Computational",
                f"(n = {n_comp:,} papers)",
                args.top_n, PALETTE["comp_or_hybrid"], comp_status)

    fig.suptitle("Most studied species by research tradition", fontsize=11, fontweight="bold", y=0.975)
    fig.text(0.5, 0.01,
             "Markers compare rank against the other tradition: ↑ ranked higher here, ↓ ranked lower here, - same rank, * absent in the other tradition",
             ha="center", va="bottom", fontsize=7.1, color="#555555")
    fig.subplots_adjust(left=0.15, right=0.98, top=0.80, bottom=0.16, wspace=0.82)

    out_pdf = out_dir / "top_species_dual.pdf"
    out_png = out_dir / "top_species_dual.png"
    save_both(fig, out_pdf, out_png, tight=False)

    assert out_pdf.exists() and out_pdf.stat().st_size > 0
    assert out_png.exists() and out_png.stat().st_size > 0
    print(f"Wrote: {out_pdf}")
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
