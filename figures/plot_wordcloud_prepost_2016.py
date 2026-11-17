#!/usr/bin/env python3
"""
Word-cloud comparison: before vs. after 2000.

Text normalization pipeline:
  1. Lowercase + tokenize (alpha tokens ≥ 3 chars)
  2. Remove a broad domain-specific + generic stopword list
  3. Lemmatize with NLTK WordNet (POS-aware); falls back to simple suffix rules
  4. Re-filter stopwords after lemmatization (catches inflected forms)
  5. Build frequency dict → WordCloud
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from _plot_utils import LoadSpec, load_df, canonicalize
from _plot_style import apply_style, save_both

SPLIT_YEAR = 2000
LAG_START  = 2025

# ── Domain + generic stopwords ─────────────────────────────────────────────────
# Includes inflected variants; post-lemmatization re-filtering catches the rest.
BASE_STOPWORDS: set[str] = {
    # Articles / conjunctions / prepositions
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
    "have", "has", "had", "there", "here", "many", "only", "same",
    "several", "often", "given", "through",
    # Generic academic / methodological
    "using", "use", "used", "uses", "based", "study", "studies", "studied",
    "analysis", "analyses", "analyzed", "analyse", "analyze",
    "result", "results", "data", "method", "methods", "approach", "approaches",
    "model", "models", "proposed", "present", "paper", "work", "show", "shown",
    "found", "find", "finding", "findings", "indicate", "indicates", "suggest",
    "suggests", "compared", "compare", "comparison", "provide", "provides",
    "provided", "demonstrate", "demonstrates", "demonstrate", "evaluate",
    "evaluation", "performance", "high", "low", "large", "small", "new",
    "novel", "different", "similar", "significant", "significantly",
    "two", "three", "one", "first", "second", "third", "number", "set",
    "across", "overall", "respectively", "well", "however", "despite",
    "note", "notes", "known", "including", "included", "particular",
    "type", "types", "class", "classes", "case", "cases",
    "process", "function", "system", "response", "effect", "information",
    "time", "level", "change", "condition", "group", "area", "range",
    "structure", "variation", "investigate", "abstract",
    # Generic bioacoustics / animal communication domain noise
    "animal", "animals", "communication", "vocal", "vocalization", "vocalizations",
    "call", "calls", "signal", "signals", "sound", "sounds",
    "acoustic", "acoustics", "song", "songs", "singing",
    "recording", "recordings", "recorded", "record",
    "species", "individual", "individuals", "population", "populations",
    "male", "female", "males", "females",
    "behavior", "behaviour", "behavioral", "behavioural",
    "field", "laboratory", "lab", "experiment", "experimental", "experiments",
    "frequency", "frequencies", "temporal", "spectral", "amplitude",
    "sequence", "sequences", "pattern", "patterns",
    # Generic computational / ML noise
    "feature", "features", "training", "trained", "test", "testing",
    "dataset", "corpus", "classifier", "classification", "network", "networks",
    "learning", "deep", "neural", "layer", "layers", "vector", "vectors",
    "audio", "waveform", "spectrogram", "mel", "mfcc",
    "accuracy", "precision", "recall", "score", "loss",
    # Lemmatized forms that still slip through
    "study", "paper", "work", "method", "approach", "model", "result",
    "use", "base", "datum", "analysis", "detect", "detection",
}


def _get_stopwords() -> set[str]:
    stopwords = set(BASE_STOPWORDS)
    try:
        import nltk
        from nltk.corpus import stopwords as nltk_stopwords

        try:
            words = nltk_stopwords.words("english")
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = nltk_stopwords.words("english")

        stopwords.update(w.lower() for w in words)
    except Exception:
        pass
    return stopwords


STOPWORDS = _get_stopwords()

TOKEN_RE = re.compile(r"[a-z]{3,}")


def _get_text(df: pd.DataFrame) -> str:
    parts: list[str] = []
    for col in ["title", "summary", "abstract_text"]:
        if col in df.columns:
            for v in df[col].dropna():
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())
    return "\n".join(parts)


def _lemmatize_tokens(tokens: list[str]) -> list[str]:
    """
    POS-aware WordNet lemmatization ONLY — no Porter stemmer.
    Porter produces aggressive, unreadable truncations (whal, mal, femal, analyz)
    which are unsuitable for display.  WordNet lemmatization is conservative:
    it only removes inflectional suffixes when the result is a known dictionary word.
    Fallback applies minimal, safe plural/verb rules only.
    """
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import wordnet

        for resource in ("averaged_perceptron_tagger", "wordnet", "omw-1.4"):
            try:
                nltk.data.find(
                    f"taggers/{resource}" if "tagger" in resource else f"corpora/{resource}"
                )
            except LookupError:
                nltk.download(resource, quiet=True)

        def _penn_to_wn(tag: str) -> str:
            if tag.startswith("J"): return wordnet.ADJ
            if tag.startswith("V"): return wordnet.VERB
            if tag.startswith("R"): return wordnet.ADV
            return wordnet.NOUN

        lem = WordNetLemmatizer()
        tagged = nltk.pos_tag(tokens)
        return [lem.lemmatize(w, _penn_to_wn(t)) for w, t in tagged]

    except Exception:
        # Conservative fallback: only safe, reversible plural/verb endings.
        # Do NOT apply aggressive suffix stripping (no Porter-style truncation).
        out: list[str] = []
        for w in tokens:
            if w.endswith("ies") and len(w) > 5:
                w = w[:-3] + "y"           # butterflies -> butterfly
            elif w.endswith("ves") and len(w) > 5:
                w = w[:-3] + "f"           # wolves -> wolf
            elif w.endswith("ing") and len(w) > 7:
                # Only strip -ing if remainder is plausibly a word (>= 4 chars)
                stem = w[:-3]
                if len(stem) >= 4:
                    w = stem
            elif w.endswith("ed") and len(w) > 5:
                stem = w[:-2]
                if len(stem) >= 4:
                    w = stem
            elif w.endswith("s") and len(w) > 4 and not w.endswith("ss"):
                w = w[:-1]
            out.append(w)
        return out


def _normalize_tokens(text: str) -> list[str]:
    assert isinstance(text, str)
    raw = TOKEN_RE.findall(text.lower())
    raw = [w for w in raw if w not in STOPWORDS]
    lemmatized = _lemmatize_tokens(raw)
    # Second pass after lemmatization (catches inflected forms reduced to stopword roots)
    canonicalized = [canonicalize(w) for w in lemmatized]
    return [w for w in canonicalized if w not in STOPWORDS and len(w) >= 3]


def _make_wordcloud(tokens: list[str]):
    try:
        from wordcloud import WordCloud
    except Exception as e:
        raise RuntimeError("Missing dependency 'wordcloud'. Install with: pip install wordcloud") from e

    freqs: dict[str, int] = {}
    for w in tokens:
        display = w.capitalize()   # e.g. "whale" -> "Whale" for readability
        freqs[display] = freqs.get(display, 0) + 1

    return WordCloud(
        width=1600,
        height=900,
        background_color="white",
        max_words=150,
        collocations=False,
        prefer_horizontal=0.85,
        colormap="viridis",
    ).generate_from_frequencies(freqs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   required=True)
    ap.add_argument("--year_min", type=int, default=1950)
    ap.add_argument("--year_max", type=int, default=2026)
    ap.add_argument("--out_dir",  default="output/wordclouds_prepost_2000")
    args = ap.parse_args()

    apply_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(LoadSpec(config=args.config, year_min=args.year_min, year_max=args.year_max))
    df = df.copy()
    df["year"] = df["year"].astype(int)
    df = df[df["year"] < LAG_START].copy()   # exclude partial years

    pre  = df[df["year"] < SPLIT_YEAR]
    post = df[df["year"] >= SPLIT_YEAR]
    assert len(pre) > 0 and len(post) > 0

    configs = [
        ("pre2000",  pre,  f"Dominant terms — before {SPLIT_YEAR}"),
        ("post2000", post, f"Dominant terms — {SPLIT_YEAR} and after"),
    ]

    for label, d, title in configs:
        text = _get_text(d)
        toks = _normalize_tokens(text)
        assert len(toks) > 100, f"Too few tokens for {label}: {len(toks)}"

        wc = _make_wordcloud(toks)

        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        n_papers = len(d)
        ax.set_title(f"{title}\n(n = {n_papers:,} papers)", fontsize=11, fontweight="bold", pad=8)

        out_pdf = out_dir / f"wordcloud_{label}.pdf"
        out_png = out_dir / f"wordcloud_{label}.png"
        save_both(fig, out_pdf, out_png, tight=True)

        assert out_pdf.exists() and out_pdf.stat().st_size > 0
        assert out_png.exists() and out_png.stat().st_size > 0
        print(f"Wrote: {out_pdf}")
        print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
