#!/usr/bin/env python3
"""Top communication features per tradition — dual lollipop chart (% share)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from _plot_utils import (
    LoadSpec,
    add_yaxis_status_markers,
    load_df,
    relative_rank_status_for_labels,
    value_counts_list,
)
from _plot_style import apply_style, save_both, PALETTE


def _norm_feature_label(x: str) -> str:
    s = str(x).strip()
    mapping = {
        "vocal auditory channel": "Auditory/Acoustic Channel",
        "acoustic auditory channel": "Auditory/Acoustic Channel",
    }
    return mapping.get(s.lower(), s)


def _topk_pct(df, group: str, col: str, k: int):
    dfg = df[df["group2"] == group]
    assert len(dfg) > 0, f"No rows for group2={group}"
    vc = value_counts_list(dfg, col, normalizer=_norm_feature_label)
    assert len(vc) > 0, f"No values in {col} for group2={group}"
    total = int(vc.sum())
    assert total > 0
    pct = (vc / total) * 100.0
    return pct.head(k).sort_values(ascending=True)


def _trend_statuses(df, group: str, labels: list[str]) -> dict[str, str]:
    focal = value_counts_list(df[df["group2"] == group], "features", normalizer=_norm_feature_label)
    other_group = "comp_or_hybrid" if group == "biological" else "biological"
    other = value_counts_list(df[df["group2"] == other_group], "features", normalizer=_norm_feature_label)
    return relative_rank_status_for_labels(
        focal.index.tolist(),
        other.index.tolist(),
        labels,
    )


def _lollipop(ax, series, title: str, subtitle: str, color: str, statuses: dict[str, str]) -> None:
    y_pos = range(len(series))
    ax.hlines(list(y_pos), xmin=0, xmax=series.values, color=color,
              linewidth=1.4, alpha=0.7)
    ax.scatter(series.values, list(y_pos), color=color, s=42, zorder=3)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(series.index.tolist(), fontsize=7.9)
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.set_xlabel("Share of feature mentions (%)")
    ax.xaxis.grid(True, linewidth=0.4, alpha=0.45, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    add_yaxis_status_markers(ax, statuses)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   required=True)
    ap.add_argument("--year_min", type=int, default=1950)
    ap.add_argument("--year_max", type=int, default=2026)
    ap.add_argument("--k",        type=int, default=12)
    ap.add_argument("--out_dir",  default="output/features")
    args = ap.parse_args()

    apply_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(LoadSpec(config=args.config, year_min=args.year_min, year_max=args.year_max))

    n_bio  = int((df["group2"] == "biological").sum())
    n_comp = int((df["group2"] == "comp_or_hybrid").sum())

    bio  = _topk_pct(df, "biological",    "features", args.k)
    comp = _topk_pct(df, "comp_or_hybrid","features", args.k)

    bio_status  = _trend_statuses(df, "biological", bio.index.tolist())
    comp_status = _trend_statuses(df, "comp_or_hybrid", comp.index.tolist())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))

    _lollipop(ax1, bio,  "Biological",
              f"(n = {n_bio:,} papers)",  PALETTE["biological"], bio_status)
    _lollipop(ax2, comp, "Computational",
              f"(n = {n_comp:,} papers)", PALETTE["comp_or_hybrid"], comp_status)

    fig.suptitle("Most analysed communication features by research tradition",
                 fontsize=11, fontweight="bold", y=0.975)
    fig.text(0.5, 0.01,
             "Markers compare rank against the other tradition: ↑ ranked higher here, ↓ ranked lower here, - same rank, * absent in the other tradition",
             ha="center", va="bottom", fontsize=7.1, color="#555555")
    fig.subplots_adjust(left=0.15, right=0.98, top=0.79, bottom=0.16, wspace=0.80)

    out_pdf = out_dir / "top_features_dual_pct.pdf"
    out_png = out_dir / "top_features_dual_pct.png"
    save_both(fig, out_pdf, out_png, tight=False)

    assert out_pdf.exists() and out_pdf.stat().st_size > 0
    assert out_png.exists() and out_png.stat().st_size > 0
    print(f"Wrote: {out_pdf}")
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
