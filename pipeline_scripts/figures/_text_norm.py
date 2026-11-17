# _text_norm.py  — NEW FILE, create in the figures/ directory

from __future__ import annotations
import re
import unicodedata

# ---------------------------------------------------------------------------
# Basic surface normalisation shared by species, wordcloud, and TF-IDF plots
# ---------------------------------------------------------------------------

def _unicode_clean(s: str) -> str:
    """Decompose accents, then re-encode as ASCII where possible."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def normalize_species_label(raw: str) -> str:
    """
    Canonicalize a species string for counting / display.

    Pipeline:
      1. Unicode → ASCII
      2. Strip surrounding whitespace
      3. Collapse internal whitespace runs to single space
      4. Lower-case
      5. Strip trailing 's' for simple English plurals
         (bats→bat, dolphins→dolphin; but 'canis lupus' left intact)
      6. Title-case the result for display
    """
    assert isinstance(raw, str), f"Expected str, got {type(raw)}"
    s = _unicode_clean(raw).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.lower()

    # Remove trailing plural 's' for single-word tokens only
    # (avoids mangling binomials like "Tursiops truncatus")
    tokens = s.split()
    if len(tokens) == 1 and tokens[0].endswith("s") and len(tokens[0]) > 3:
        tokens[0] = tokens[0][:-1]
    s = " ".join(tokens)

    return s.title()   # "killer whale", "bat" → "Killer Whale", "Bat"


# Domain-specific stopwords for an animal bioacoustics / animal comms survey
DOMAIN_STOPWORDS: frozenset[str] = frozenset({
    # generic academic boilerplate
    "the","a","an","and","or","of","to","in","for","with","on","by","from","at","as",
    "is","are","was","were","be","been","being","this","that","these","those",
    "we","our","their","they","it","its","into","during","which","have","has","had",
    "than","more","also","between","among","across","within","without","such","both",
    # methodology boilerplate
    "using","use","used","based","study","studies","analysis","results","data",
    "method","methods","approach","approaches","proposed","present","new","show",
    "shown","can","may","model","models","paper","work","system","systems","number",
    "different","high","low","large","small","significant","significantly","found",
    "two","three","one","first","second","type","types","set","sets","case","cases",
    # animal-comms domain-generic (too frequent to be informative)
    "animal","animals","communication","vocal","vocalisation","vocalization",
    "calls","call","signals","signal","sound","sounds","acoustic","acoustics",
    "species","behavior","behaviour","song","songs","note","notes","sequence",
    "recorded","recording","recordings","produced","production","frequency",
    "time","temporal","rate","pattern","patterns",
})


def normalize_text_tokens(text: str) -> list[str]:
    """
    Tokenize and clean free text (titles / abstracts) for wordcloud / TF-IDF.

    Returns a list of normalized tokens (already stemmed if nltk available).
    """
    assert isinstance(text, str)
    t = _unicode_clean(text).lower()
    toks = re.findall(r"[a-z]{3,}", t)
    toks = [w for w in toks if w not in DOMAIN_STOPWORDS]

    # Optional Porter stemming (silently skipped if nltk absent)
    try:
        from nltk.stem import PorterStemmer
        ps = PorterStemmer()
        toks = [ps.stem(w) for w in toks]
    except Exception:
        pass

    return toks
