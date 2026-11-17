#!/usr/bin/env python3
"""
TF-IDF term analysis: before vs. after 2000.

Presents THREE panels:
  (left)   Top terms pre-2000        — horizontal bars
  (right)  Top terms post-2000       — horizontal bars
  (bottom) Δ TF-IDF (post − pre)     — diverging bar chart highlighting
           terms that rose or fell most sharply across the split year.

Normalization:
  - POS-aware lemmatization via NLTK WordNet (fallback: suffix rules)
  - Comprehensive domain + generic stopword list (shared with wordcloud script)
  - Unigrams only (cleaner; bigrams optional via --ngrams 2)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from _plot_utils import LoadSpec, load_df, canonicalize
from _plot_style import apply_style, save_both, PALETTE

SPLIT_YEAR = 2000
LAG_START  = 2025

# ── Shared stopword set (same as wordcloud script) ─────────────────────────────
STOPWORDS: set[str] = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "on",
    "by", "from", "at", "as", "is", "are", "was", "were", "be", "been",
    "being", "this", "that", "these", "those", "we", "our", "their", "they",
    "it", "its", "into", "during", "which", "who", "what", "where", "when",
    "how", "all", "also", "both", "each", "few", "more", "most", "other",
    "some", "such", "than", "then", "too", "very", "can", "will", "may",
    "might", "would", "could", "should", "not", "but", "yet", "nor",
    "while", "although", "however", "therefore", "thus", "hence", "whether",
    "within", "without", "between", "among", "about", "above", "below",
    "after", "before", "since", "until", "towards", "against",
    "using", "use", "used", "uses", "based", "study", "studies", "studied",
    "analysis", "analyses", "analyzed", "result", "results", "data",
    "method", "methods", "approach", "approaches", "model", "models",
    "proposed", "present", "paper", "work", "show", "shown", "found",
    "find", "finding", "findings", "indicate", "suggests", "compared",
    "compare", "comparison", "provide", "provides", "demonstrate",
    "evaluate", "evaluation", "performance", "high", "low", "large",
    "small", "new", "novel", "different", "similar", "significant",
    "significantly", "two", "three", "one", "first", "second", "third",
    "number", "set", "across", "overall", "respectively", "well",
    "despite", "note", "known", "including", "particular",
    "type", "types", "class", "classes", "case", "cases",
    # domain noise
    "animal", "animals", "communication", "vocal", "vocalization", "vocalizations",
    "call", "calls", "signal", "signals", "sound", "sounds",
    "acoustic", "acoustics", "song", "songs", "singing",
    "recording", "recordings", "recorded", "record",
    "species", "individual", "individuals", "population", "populations",
    "behavior", "behaviour", "behavioral", "behavioural",
    "field", "laboratory", "lab", "experiment", "experimental", "experiments",
    "frequency", "frequencies", "temporal", "spectral", "amplitude",
    "sequence", "sequences", "pattern", "patterns",
    "feature", "features", "training", "trained", "test", "testing",
    "dataset", "corpus", "classifier", "classification", "network", "networks",
    "learning", "deep", "neural", "layer", "layers", "vector", "vectors",
    "audio", "waveform", "spectrogram",
    "accuracy", "precision", "recall", "score", "loss",
    # lemmatized roots that still slip through
    "study", "paper", "work", "method", "approach", "model", "result",
    "use", "base", "datum",
}


def _join_text_row(row: pd.Series) -> str:
    parts: list[str] = []
    for col in ["title", "summary", "abstract_text"]:
        v = row.get(col)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return " ".join(parts)


def _build_corpus(df: pd.DataFrame) -> list[str]:
    if len(df) == 0:
        return []
    docs = df.apply(_join_text_row, axis=1).tolist()
    return [d for d in docs if isinstance(d, str) and d.strip()]


def _lemmatize_doc(text: str, lem, penn_to_wn, tagger) -> str:
    """Lemmatize a single document string (used inside the vectorizer analyzer)."""
    import re
    tokens = re.findall(r"[a-z]{3,}", text.lower())
    tokens = [w for w in tokens if w not in STOPWORDS]
    tagged = tagger(tokens)
    lemmas = [lem.lemmatize(w, penn_to_wn(t)) for w, t in tagged]
    # second-pass stopword filter
    return " ".join(w for w in lemmas if w not in STOPWORDS and len(w) >= 3)


def _build_analyzer(ngram_max: int):
    """
    Returns a sklearn-compatible analyzer callable that applies
    lemmatization before n-gram extraction.
    Falls back to simple suffix stripping if NLTK is unavailable.
    """
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import wordnet

        for resource in ("averaged_perceptron_tagger", "wordnet", "omw-1.4"):
            try:
                nltk.data.find(f"taggers/{resource}" if "tagger" in resource else f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

        def _penn_to_wn(tag: str) -> str:
            if tag.startswith("J"): return wordnet.ADJ
            if tag.startswith("V"): return wordnet.VERB
            if tag.startswith("R"): return wordnet.ADV
            return wordnet.NOUN

        lem = WordNetLemmatizer()

        def analyzer(doc: str) -> list[str]:
            import re
            from sklearn.feature_extraction.text import CountVectorizer
            tokens = re.findall(r"[a-z]{3,}", doc.lower())
            tokens = [w for w in tokens if w not in STOPWORDS]
            tagged = nltk.pos_tag(tokens)
            lemmas = [lem.lemmatize(w, _penn_to_wn(t)) for w, t in tagged]
            lemmas = [canonicalize(w) for w in lemmas]
            lemmas = [w for w in lemmas if w not in STOPWORDS and len(w) >= 3]
            # produce n-grams
            out: list[str] = []
            for n in range(1, ngram_max + 1):
                out.extend(" ".join(lemmas[i:i+n]) for i in range(len(lemmas) - n + 1))
            return out

        return analyzer

    except Exception:
        # Minimal fallback analyzer
        import re as _re

        def analyzer(doc: str) -> list[str]:  # type: ignore[misc]
            tokens = _re.findall(r"[a-z]{3,}", doc.lower())
            tokens = [canonicalize(w) for w in tokens if w not in STOPWORDS]
            tokens = [w for w in tokens if w not in STOPWORDS]
            out: list[str] = []
            for n in range(1, ngram_max + 1):
                out.extend(" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
            return out

        return analyzer


def _top_terms(docs: list[str], top_n: int, ngram_max: int = 1) -> dict[str, float]:
    assert len(docs) > 50, f"Too few docs for TF-IDF: {len(docs)}"

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as e:
        raise RuntimeError("Missing 'scikit-learn'. Install: pip install scikit-learn") from e

    analyzer = _build_analyzer(ngram_max)

    vec = TfidfVectorizer(
        analyzer=analyzer,
        min_df=5,
        max_df=0.70,
    )
    X = vec.fit_transform(docs)
    assert X.shape[1] > 0, "TF-IDF vocabulary empty — lower min_df or adjust stopwords."

    scores = np.asarray(X.mean(axis=0)).ravel()
    terms  = np.array(vec.get_feature_names_out())

    idx = np.argsort(scores)[::-1][:top_n]
    return {str(terms[i]): float(scores[i]) for i in idx}


def _plot_side(ax, items: list[tuple[str, float]], title: str, color: str) -> None:
    terms = [t.capitalize() for t, _ in items][::-1]
    vals  = [v for _, v in items][::-1]
    y_pos = range(len(terms))

    ax.hlines(y=list(y_pos), xmin=0, xmax=vals, color=color, linewidth=1.3, alpha=0.75)
    ax.scatter(vals, list(y_pos), color=color, s=35, zorder=3)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(terms, fontsize=8.0)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Mean TF-IDF")
    ax.xaxis.grid(True, linewidth=0.4, alpha=0.45, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def _plot_delta(ax, delta: pd.Series, top_n: int) -> None:
    """
    Diverging bar: top_n rising terms + top_n falling terms, sorted by delta.
    """
    rising  = delta.nlargest(top_n)
    falling = delta.nsmallest(top_n)
    combined = pd.concat([falling, rising]).sort_values()

    colors = [PALETTE["post"] if v >= 0 else PALETTE["pre"] for v in combined.values]

    y_pos = range(len(combined))
    ax.barh(list(y_pos), combined.values, color=colors, edgecolor="none", height=0.7)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([t.capitalize() for t in combined.index.tolist()], fontsize=8.0)
    ax.set_title(f"Δ TF-IDF  (post {SPLIT_YEAR} − pre {SPLIT_YEAR})", fontsize=10)
    ax.set_xlabel("Mean TF-IDF change")
    ax.xaxis.grid(True, linewidth=0.4, alpha=0.45, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    patch_rise = mpatches.Patch(color=PALETTE["post"],  label=f"Rising  (≥ {SPLIT_YEAR})")
    patch_fall = mpatches.Patch(color=PALETTE["pre"],   label=f"Falling (< {SPLIT_YEAR})")
    ax.legend(handles=[patch_rise, patch_fall], fontsize=7.8, frameon=False, loc="lower right")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   required=True)
    ap.add_argument("--year_min", type=int, default=1950)
    ap.add_argument("--year_max", type=int, default=2026)
    ap.add_argument("--top_n",    type=int, default=20,
                    help="Top-N terms per period (also used for delta panel)")
    ap.add_argument("--ngrams",   type=int, default=1,
                    help="Max n-gram length (1=unigrams only, 2=bigrams included)")
    ap.add_argument("--out_dir",  default="output/tfidf_prepost_2000")
    args = ap.parse_args()
    assert args.top_n >= 10

    apply_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(LoadSpec(config=args.config, year_min=args.year_min, year_max=args.year_max))
    df = df.copy()
    df["year"] = df["year"].astype(int)
    df = df[df["year"] < LAG_START].copy()
    assert len(df) > 0, "No rows after excluding lag years."

    pre  = df[df["year"] < SPLIT_YEAR]
    post = df[df["year"] >= SPLIT_YEAR]
    assert len(pre) > 50 and len(post) > 50, \
        f"Too few docs: pre={len(pre)} post={len(post)}"

    docs_pre  = _build_corpus(pre)
    docs_post = _build_corpus(post)

    scores_pre  = _top_terms(docs_pre,  args.top_n * 3, args.ngrams)  # fetch wider for delta
    scores_post = _top_terms(docs_post, args.top_n * 3, args.ngrams)

    # Top-N for side panels (sorted by score descending)
    top_pre  = sorted(scores_pre.items(),  key=lambda x: -x[1])[:args.top_n]
    top_post = sorted(scores_post.items(), key=lambda x: -x[1])[:args.top_n]

    # Delta: terms present in both corpora
    common = set(scores_pre) & set(scores_post)
    delta = pd.Series(
        {t: scores_post[t] - scores_pre[t] for t in common}
    ).sort_values()

    n_pre  = len(pre)
    n_post = len(post)
    delta_n = min(12, args.top_n // 2)

    # Figure 1: side-by-side top-N term panels
    fig1, (ax_pre, ax_post) = plt.subplots(1, 2, figsize=(8.8, 5.4), sharey=False)
    _plot_side(ax_pre,  top_pre,
               f"Top terms — before {SPLIT_YEAR}\n(n = {n_pre:,})",  PALETTE["pre"])
    _plot_side(ax_post, top_post,
               f"Top terms — {SPLIT_YEAR} and after\n(n = {n_post:,})", PALETTE["post"])
    fig1.suptitle("Dominant terms in animal communication research across 2000",
                  fontsize=11, fontweight="bold")
    out1_pdf = out_dir / f"tfidf_top_terms_{SPLIT_YEAR}.pdf"
    out1_png = out_dir / f"tfidf_top_terms_{SPLIT_YEAR}.png"
    save_both(fig1, out1_pdf, out1_png, tight=True)
    assert out1_pdf.exists() and out1_pdf.stat().st_size > 0
    print(f"Wrote: {out1_pdf}")
    print(f"Wrote: {out1_png}")

    # Figure 2: diverging delta panel
    fig2, ax_delta = plt.subplots(figsize=(7.2, 5.2))
    _plot_delta(ax_delta, delta, top_n=delta_n)
    fig2.suptitle("Lexical shifts across 2000 — rising and falling terms",
                  fontsize=11, fontweight="bold")
    out2_pdf = out_dir / f"tfidf_delta_{SPLIT_YEAR}.pdf"
    out2_png = out_dir / f"tfidf_delta_{SPLIT_YEAR}.png"
    save_both(fig2, out2_pdf, out2_png, tight=True)
    assert out2_pdf.exists() and out2_pdf.stat().st_size > 0
    print(f"Wrote: {out2_pdf}")
    print(f"Wrote: {out2_png}")



if __name__ == "__main__":
    main()
