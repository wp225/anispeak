#!/usr/bin/env python3
"""
Compact pie charts for application domains, communication modalities, and research
settings.

The script preserves the per-facet exports and also writes a combined two-panel
figure for domains + research settings.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _plot_utils import LoadSpec, load_df, explode_list_col, assert_listlike
from _plot_style import apply_style, save_both, COLORS_CYCLE


def _norm_label(x: str) -> str:
    assert isinstance(x, str)
    s  = x.strip()
    if not s:
        return "Other"
    sl = s.lower()
    nullish = {
        "na", "n/a", "none", "null", "unknown", "unk", "undefined",
        "not_applicable", "not applicable", "not-applicable", "n.a.",
        "notapplicable",
    }
    if sl in nullish or sl == "other" or sl.startswith("other ") \
            or sl in {"misc", "miscellaneous"}:
        return "Other"
    return s


def _value_counts_normalized(s: pd.Series) -> pd.Series:
    assert isinstance(s, pd.Series)
    if len(s) == 0:
        return pd.Series([], dtype=int)
    s2 = s.map(lambda v: _norm_label(v) if isinstance(v, str) else "Other")
    return s2.value_counts()


def _format_labels(top: pd.Series, total: int) -> list[str]:
    return [f"{k} ({v / total * 100:.1f}%)" for k, v in top.items()]


def pie_panel(ax, counts: pd.Series, title: str, top_n: int = 10, legend_cols: int = 2) -> None:
    counts = counts[counts > 0]
    assert len(counts) > 0, f"No counts for {title}"

    top  = counts.head(top_n).copy()
    rest = int(counts.iloc[top_n:].sum())
    if rest > 0:
        if "Other" in top.index:
            top.loc["Other"] += rest
        else:
            top = pd.concat([top, pd.Series({"Other": rest})])

    # Put "Other" last
    if "Other" in top.index:
        other_val = top.pop("Other")
        top["Other"] = other_val

    total = int(top.sum())
    n_slices = len(top)

    # Use the shared palette cycle; grey for "Other"
    colors = [COLORS_CYCLE[i % len(COLORS_CYCLE)] for i in range(n_slices)]
    if "Other" in top.index:
        colors[list(top.index).index("Other")] = "#aaaaaa"

    wedges, _ = ax.pie(
        top.values,
        colors=colors,
        startangle=90,
        counterclock=False,
        radius=0.93,
        wedgeprops={"linewidth": 0.7, "edgecolor": "white"},
    )
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=6)
    ax.set(aspect="equal")

    labels = _format_labels(top, total)
    ax.legend(
        wedges,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
        fontsize=8.6,
        handlelength=1.0,
        handletextpad=0.5,
        columnspacing=1.0,
        labelspacing=0.45,
        borderaxespad=0.0,
        ncol=min(legend_cols, len(labels)),
    )


def make_one(df: pd.DataFrame, col: str, title: str, out_dir: Path, top_n: int) -> None:
    assert_listlike(df, col)
    raw = explode_list_col(df, col)
    vc  = _value_counts_normalized(raw)
    assert len(vc) > 0, f"No values for {col}"

    n_total = len(df)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    pie_panel(ax, vc, f"{title}\n(n = {n_total:,})", top_n=top_n, legend_cols=2)

    out_pdf = out_dir / f"{col}_donut.pdf"
    out_png = out_dir / f"{col}_donut.png"
    save_both(fig, out_pdf, out_png, tight=True)

    assert out_pdf.exists() and out_pdf.stat().st_size > 0
    assert out_png.exists() and out_png.stat().st_size > 0
    print(f"Wrote: {out_pdf}")
    print(f"Wrote: {out_png}")


def make_pair(
    df: pd.DataFrame,
    left_col: str,
    left_title: str,
    right_col: str,
    right_title: str,
    out_dir: Path,
    top_n: int,
) -> None:
    assert_listlike(df, left_col)
    assert_listlike(df, right_col)

    left_vc = _value_counts_normalized(explode_list_col(df, left_col))
    right_vc = _value_counts_normalized(explode_list_col(df, right_col))
    assert len(left_vc) > 0, f"No values for {left_col}"
    assert len(right_vc) > 0, f"No values for {right_col}"

    n_total = len(df)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.2))
    pie_panel(axes[0], left_vc, f"{left_title}\n(n = {n_total:,})", top_n=top_n, legend_cols=2)
    pie_panel(axes[1], right_vc, f"{right_title}\n(n = {n_total:,})", top_n=top_n, legend_cols=2)
    fig.subplots_adjust(wspace=0.36, bottom=0.26)

    out_pdf = out_dir / "domains_contexts_compact.pdf"
    out_png = out_dir / "domains_contexts_compact.png"
    save_both(fig, out_pdf, out_png, tight=True)

    assert out_pdf.exists() and out_pdf.stat().st_size > 0
    assert out_png.exists() and out_png.stat().st_size > 0
    print(f"Wrote: {out_pdf}")
    print(f"Wrote: {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   required=True)
    ap.add_argument("--year_min", type=int, default=1950)
    ap.add_argument("--year_max", type=int, default=2026)
    ap.add_argument("--top_n",    type=int, default=10)
    ap.add_argument("--out_dir",  default="output/donuts")
    args = ap.parse_args()

    apply_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(LoadSpec(config=args.config, year_min=args.year_min, year_max=args.year_max))

    make_pair(df, "domains", "Application domains", "contexts", "Research settings", out_dir, args.top_n)
    make_one(df, "domains",    "Application domains",      out_dir, args.top_n)
    make_one(df, "modalities", "Communication modalities", out_dir, args.top_n)
    make_one(df, "contexts",   "Research settings",        out_dir, args.top_n)


if __name__ == "__main__":
    main()
