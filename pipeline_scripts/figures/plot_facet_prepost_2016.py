#!/usr/bin/env python3
"""
Facet distributions — before vs. after 2000.

Uses a DIVERGING DOT-PLOT (connected dumbbell chart) rather than two separate
bar panels. Both periods share the same y-axis, making the shift for each
category immediately legible.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from _plot_utils import LoadSpec, load_df, explode_list_col, assert_listlike
from _plot_style import apply_style, save_both, PALETTE

SPLIT_YEAR = 2000
LAG_START  = 2025


def _norm_label(x: str) -> str:
    assert isinstance(x, str)
    s  = x.strip()
    if not s:
        return "Other"
    sl = s.lower()
    if sl in {
        "na", "n/a", "none", "null", "unknown", "undefined",
        "not_applicable", "not applicable", "not-applicable", "notapplicable",
        "other", "misc", "miscellaneous",
    } or sl.startswith("other "):
        return "Other"
    return s


def _facet_counts(df: pd.DataFrame, col: str) -> pd.Series:
    assert_listlike(df, col)
    raw = explode_list_col(df, col)
    s   = raw.map(lambda v: _norm_label(v) if isinstance(v, str) else "Other")
    vc  = s.value_counts()
    assert len(vc) > 0, f"No values for {col}"
    return vc


def _topk_with_other(vc: pd.Series, k: int) -> pd.Series:
    assert k >= 1
    top  = vc.head(k)
    rest = int(vc.iloc[k:].sum())
    if rest > 0:
        if "Other" in top.index:
            top = top.copy()
            top.loc["Other"] += rest
        else:
            top = pd.concat([top, pd.Series({"Other": rest})])
    return top


def _dumbbell(ax, cats: list[str], vals_pre: list[float], vals_post: list[float],
              n_pre: int, n_post: int) -> None:
    """
    Horizontal dumbbell / connected dot plot.
    Pre = slate-blue circle, Post = amber circle, connected by a grey line.
    """
    y_pos = range(len(cats))
    col_pre  = PALETTE["pre"]
    col_post = PALETTE["post"]

    for y, vp, vq in zip(y_pos, vals_pre, vals_post):
        ax.plot([vp, vq], [y, y], color="#bbbbbb", linewidth=1.2, zorder=1)

    ax.scatter(vals_pre,  list(y_pos), color=col_pre,  s=55, zorder=3,
               label=f"Before {SPLIT_YEAR} (n = {n_pre:,})")
    ax.scatter(vals_post, list(y_pos), color=col_post, s=55, zorder=3,
               label=f"{SPLIT_YEAR}+ (n = {n_post:,})", marker="D")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(cats, fontsize=8.2)
    ax.set_xlabel("Share of mentions (%)")
    ax.xaxis.grid(True, linewidth=0.4, alpha=0.45, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   required=True)
    ap.add_argument("--year_min", type=int, default=1950)
    ap.add_argument("--year_max", type=int, default=2026)
    ap.add_argument("--top_n",    type=int, default=10)
    ap.add_argument("--out_dir",  default="output/facets_prepost_2000")
    args = ap.parse_args()
    assert args.top_n >= 3

    apply_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(LoadSpec(config=args.config, year_min=args.year_min, year_max=args.year_max))
    df = df.copy()
    df["year"] = df["year"].astype(int)
    df_full = df[df["year"] < LAG_START].copy()

    pre  = df_full[df_full["year"] < SPLIT_YEAR]
    post = df_full[df_full["year"] >= SPLIT_YEAR]
    assert len(pre) > 0 and len(post) > 0, "Empty pre/post split."

    n_pre  = len(pre)
    n_post = len(post)

    for col, title in [
        ("domains",   "Application domains"),
        ("contexts",  "Research settings"),
        ("modalities","Communication modalities"),
    ]:
        vc_pre  = _facet_counts(pre,  col)
        vc_post = _facet_counts(post, col)

        # Union of top-k categories, ranked by combined frequency
        top_pre  = _topk_with_other(vc_pre,  args.top_n)
        top_post = _topk_with_other(vc_post, args.top_n)
        combined = (vc_pre.add(vc_post, fill_value=0)).sort_values(ascending=False)
        cats = [c for c in combined.index if c in top_pre.index or c in top_post.index]
        cats = cats[:args.top_n + 2]   # guard against explosion

        # Convert to % share
        pre_total  = int(vc_pre.sum())
        post_total = int(vc_post.sum())
        assert pre_total > 0 and post_total > 0

        vals_pre  = [(vc_pre.get(c, 0) / pre_total  * 100) for c in cats]
        vals_post = [(vc_post.get(c, 0) / post_total * 100) for c in cats]

        # Sort categories by post-period share for cleaner visual order
        order = sorted(range(len(cats)), key=lambda i: vals_post[i])
        cats_ord      = [cats[i]      for i in order]
        vals_pre_ord  = [vals_pre[i]  for i in order]
        vals_post_ord = [vals_post[i] for i in order]

        fig, ax = plt.subplots(figsize=(6.4, max(3.0, len(cats) * 0.38)))
        _dumbbell(ax, cats_ord, vals_pre_ord, vals_post_ord, n_pre, n_post)
        ax.set_title(f"{title}: share before vs. after {SPLIT_YEAR}",
                     fontsize=10, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8.0, frameon=False)

        fig.tight_layout()

        out_pdf = out_dir / f"{col}_prepost_{SPLIT_YEAR}_dumbbell.pdf"
        out_png = out_dir / f"{col}_prepost_{SPLIT_YEAR}_dumbbell.png"
        save_both(fig, out_pdf, out_png, tight=False)

        assert out_pdf.exists() and out_pdf.stat().st_size > 0
        assert out_png.exists() and out_png.stat().st_size > 0
        print(f"Wrote: {out_pdf}")
        print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
